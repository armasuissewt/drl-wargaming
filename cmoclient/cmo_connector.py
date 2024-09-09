# cmo_connector.py
#
# LUA based interface to Command Modern Operations
#
# Author: Giacomo Del Rio
# Creation date: 28 October 2021

import atexit
import math
import os
import socket
import tempfile
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

from cmoclient.lua_parser import LuaParser


def remove_file(fname: str):
    try:
        os.remove(fname)
    except OSError as _:
        pass


class ResponseType(Enum):
    SINGLE = 1
    LONG = 2
    XML = 3


class CMOConnectorException(Exception):

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause
        self.message = message

    def __str__(self):
        if self.cause is not None:
            return f"CMOConnectorException: {self.message}\nCause: {self.cause}"
        else:
            return f"CMOConnectorException: {self.message}\n"


class CMOConnector:
    SOCKET_TIMEOUT = 10
    MAX_CONNECT_ATTEMPTS = 5
    CONNECT_RETRY_SLEEP = 0.5
    MSG_ENCODING = 'UTF-8'
    RECV_BUFFER_SIZE = 4096
    MAX_COMMUNICATE_ATTEMPTS = 5
    COMMUNICATE_RETRY_SLEEP = 0.5
    RUN_PERIOD_POLLING_INTERVAL = 0.05
    XML_TERMINATION_STRING = "</Scenario>".encode(MSG_ENCODING)

    def __init__(self, address: Tuple = ('localhost', 7777)):
        self.sock = None
        self.address = address
        self.xml_temp_file = tempfile._get_default_tempdir() + "\\" + next(tempfile._get_candidate_names())  # noqa
        self.xml_temp_file = self.xml_temp_file.replace('\\', '\\\\')
        atexit.register(remove_file, self.xml_temp_file)

    def connect(self) -> ():
        for i in range(CMOConnector.MAX_CONNECT_ATTEMPTS):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(CMOConnector.SOCKET_TIMEOUT)
                self.sock.connect(self.address)
                return
            except OSError as e:
                if e.winerror == 10061:  # No connection could be made because the target machine actively refused it
                    print(f"CMOConnector.connect() Got {e.winerror} on attempt {i + 1}. "
                          f"Try again in {CMOConnector.CONNECT_RETRY_SLEEP} seconds...")
                    time.sleep(CMOConnector.CONNECT_RETRY_SLEEP)
                else:
                    raise CMOConnectorException("Can't connect to CMO", e)

    def disconnect(self) -> ():
        self.sock.close()
        self.sock = None

    def reconnect(self) -> ():
        self.disconnect()
        self.connect()

    def _send(self, lua_msg: str) -> ():
        try:
            self.sock.sendall(lua_msg.encode(CMOConnector.MSG_ENCODING))
        except socket.timeout as e:
            raise CMOConnectorException("Timeout on _send()", e)

    def _receive_single(self) -> str:
        try:
            chunk = self.sock.recv(CMOConnector.RECV_BUFFER_SIZE)
            return chunk.decode(encoding=CMOConnector.MSG_ENCODING)
        except socket.timeout as e:
            raise CMOConnectorException("Timeout on _receive_single()", e)

    def _receive_long(self) -> str:
        try:
            rcv_buffer = bytearray()
            paren_count = 0
            while True:
                chunk = self.sock.recv(CMOConnector.RECV_BUFFER_SIZE)
                rcv_buffer.extend(chunk)

                # Check for balanced {} to terminate
                for i in chunk:
                    if i == 123:  # {
                        paren_count += 1
                    elif i == 125:  # }
                        paren_count -= 1
                if paren_count == 0:
                    break

            return rcv_buffer.decode(encoding=CMOConnector.MSG_ENCODING)
        except socket.timeout as e:
            raise CMOConnectorException("Timeout on _receive_long()", e)

    def _receive_xml(self) -> str:
        try:
            rcv_buffer = bytearray()
            while True:
                chunk = self.sock.recv(CMOConnector.RECV_BUFFER_SIZE)
                rcv_buffer.extend(chunk)

                # Check for termination
                if chunk.endswith(CMOConnector.XML_TERMINATION_STRING):
                    break

            return rcv_buffer.decode(encoding=CMOConnector.MSG_ENCODING)
        except socket.timeout as e:
            raise CMOConnectorException("Timeout on _receive_xml()", e)

    def _communicate(self, request: str, res_type: ResponseType = ResponseType.LONG) -> str:
        for i in range(CMOConnector.MAX_COMMUNICATE_ATTEMPTS):
            try:
                self._send(request)
                if res_type == ResponseType.SINGLE:
                    return self._receive_single()
                elif res_type == ResponseType.LONG:
                    return self._receive_long()
                elif res_type == ResponseType.XML:
                    return self._receive_xml()
            except OSError as e:
                if e.winerror == 10054:  # An existing connection was forcibly closed by the remote host
                    # print(f"Communicate-wait")
                    self.reconnect()
                    time.sleep(CMOConnector.COMMUNICATE_RETRY_SLEEP)
                else:
                    raise CMOConnectorException(f"Unexpected exception on _communicate().\nLUA request: {request}", e)
            except CMOConnectorException as e:
                raise e

        raise CMOConnectorException(f"CMOConnector._communicate() Can't communicate with CMANO "
                                    f"after {CMOConnector.MAX_COMMUNICATE_ATTEMPTS} attempts.\nLUA request: {request}")

    def get_scenario(self) -> Dict:
        result = self._communicate("VP_GetScenario()")
        return CMOConnector._parse_object(result)

    def end_scenario(self) -> ():
        result = self._communicate("ScenEdit_EndScenario()", ResponseType.SINGLE)
        if result == '1':
            return
        else:
            raise CMOConnectorException(f"ScenEdit_EndScenario() returned {result}")

    def get_sides(self) -> List[Dict]:
        result = self._communicate("VP_GetSides()")
        lap = LuaParser(result)
        sides = lap.parse_array()
        return [s[1] for s in sides]

    def get_side(self, guid: str) -> Optional[Dict]:
        result = self._communicate("VP_GetSide({Side='" + guid + "'})", ResponseType.SINGLE)
        return CMOConnector._parse_object(result)

    def get_side_units(self, guid: str) -> List[Dict]:
        result = self._communicate("VP_GetSide({Side='" + guid + "'}).units")
        if result.startswith('Internal ERROR'):
            raise CMOConnectorException(f"VP_GetSide().units returned {result}")
        lap = LuaParser(result)
        sides = lap.parse_array()
        return [s[1] for s in sides]

    def get_unit(self, guid: str) -> Optional[Dict]:
        result = self._communicate("VP_GetUnit({guid='" + guid + "'})")
        unit = CMOConnector._parse_object(result)
        if unit:
            unit['latitude'] = float(unit['latitude'].replace(",", "."))
            unit['longitude'] = float(unit['longitude'].replace(",", "."))
            unit['altitude'] = float(unit['altitude'].replace(",", "."))
            unit['heading'] = float(unit['heading'].replace(",", "."))
            unit['speed'] = float(unit['speed'].replace(",", "."))
        return unit

    def get_unit_property(self, guid: str, prop: str) -> str:
        return self._communicate(f"VP_GetUnit({{guid='{guid}'}}).{prop}")

    def set_unit_property(self, guid: str, prop: str, value: str) -> str:
        return self._communicate(f"ScenEdit_SetUnit({{guid='{guid}', {prop}={value}}})")

    def get_unit_mounts(self, guid: str) -> List[Dict]:
        result = self._communicate("VP_GetUnit({guid='" + guid + "'}).mounts")
        mounts = [p[1] for p in LuaParser(result).parse_array()]
        return mounts

    def get_unit_loadout_dbid(self, guid: str) -> Optional[int]:
        result = self._communicate("VP_GetUnit({guid='" + guid + "'}).loadoutdbid")
        if result.startswith('Internal ERROR'):
            return None
        else:
            return int(result)

    def get_loadout_weapons(self, unit_guid: str, loadout_id: Optional[int] = None) -> List[Dict]:
        if loadout_id:
            result = self._communicate("ScenEdit_GetLoadout({UnitName='" + unit_guid +
                                       "',  LoadoutID='" + str(loadout_id) + "'}).weapons")
        else:
            result = self._communicate("ScenEdit_GetLoadout({UnitName='" + unit_guid + "', LoadoutID=0}).weapons")
        return [p[1] for p in LuaParser(result).parse_array()]

    def get_sensors(self, guid: str) -> List[Dict]:
        result = self._communicate("ScenEdit_SetUnit({guid='" + guid + "'}).sensors")
        return [p[1] for p in LuaParser(result).parse_array()]

    def set_unit_sensor(self, guid: str, sensor_guid: str, enabled: bool) -> bool:
        result = self._communicate(f"--script\n"
                                   f"ScenEdit_GetUnit({{guid='{guid}'}}).sensors="
                                   f"{{sensor_guid='{sensor_guid}', sensor_isactive='{enabled}'}}")
        return result == 'OK'

    def set_pose(self, guid: str, lat: float, lon: float, heading: float) -> bool:
        result = self._communicate(f"ScenEdit_SetUnit({{guid='{guid}', lat={lat}, lon={lon}, heading={heading}}})")
        return result != 'nil'

    def set_altitude(self, guid: str, altitude: Union[str, float]) -> bool:
        # https://www.matrixgames.com/forums/tm.asp?m=4805102
        # Altitude ASL in meters if number, otherwise:
        #   min, low,low1000,low2000, med,medium12000, high25000,high,high36000, max
        result = self._communicate(f"ScenEdit_SetUnit({{guid='{guid}', manualAltitude='{altitude}'}})")
        return result != 'nil'

    def set_throttle(self, guid: str, throttle: str) -> bool:
        # https://www.matrixgames.com/forums/tm.asp?m=4805102
        # Throttle in: 'loiter/creep', 'cruise', 'full', 'flank'
        result = self._communicate(f"ScenEdit_SetUnit({{guid='{guid}', manualThrottle='{throttle}'}})")
        return result != 'nil'

    def set_altitude_and_throttle(self, guid: str, altitude: Union[str, float], throttle: str) -> bool:
        result = self._communicate(f"ScenEdit_SetUnit({{guid='{guid}', manualAltitude='{altitude}', "
                                   f"manualThrottle='{throttle}'}})")
        return result != 'nil'

    def set_emcon(self, emcon_type: str, name: str, emcon: str) -> bool:
        result = self._communicate(f"ScenEdit_SetEMCON('{emcon_type}', '{name}', '{emcon}')")
        return result == "'Yes'"

    def set_doctrine(self, side: str, unit_guid: str, doctrine: Dict) -> bool:
        doct_str = ", ".join([f"{str(k)}=\"{str(doctrine[k])}\"" for k in doctrine])
        result = self._communicate(f"ScenEdit_SetDoctrine({{side=\"{side}\", unitname=\"{unit_guid}\"}}, "
                                   f"{{{doct_str}}})")
        return result.startswith('{')

    def set_side_posture(self, side_a: str, side_b: str, posture: str) -> bool:
        result = self._communicate(f"ScenEdit_SetSidePosture('{side_a}', '{side_b}', '{posture}')")
        return result == "'Yes'"

    def set_heading(self, guid: str, heading: float) -> bool:
        # heading in [0, 360)
        result = self._communicate(f"ScenEdit_SetUnit({{guid='{guid}', heading={heading}}})")
        return result != 'nil'

    def modify_heading(self, guid: str, delta: float) -> bool:
        unit = self.get_unit(guid)
        if unit:
            new_heading = unit['heading'] + delta
            if new_heading >= 360:
                new_heading = new_heading - 360
            elif new_heading < 0:
                new_heading = new_heading + 360
            return self.set_heading(guid, new_heading)
        else:
            return False

    def set_course(self, unit_guid: str, waypoint: Tuple[float, float]) -> bool:
        waypoint_str = f"{{latitude = {waypoint[0]}, longitude = {waypoint[1]}}}"
        msg = "ScenEdit_SetUnit({guid='" + unit_guid + "', course={ [1] = " + waypoint_str + "}})"
        result = self._communicate(msg)
        return result != 'nil'

    def modify_course_relative(self, unit_guid: str, delta_angle: float) -> bool:
        unit = self.get_unit(unit_guid)
        if unit:
            new_heading = unit['heading'] + delta_angle
            if new_heading >= 360:
                new_heading = new_heading - 360
            elif new_heading < 0:
                new_heading = new_heading + 360
            rad = (new_heading / 360) * math.pi * 2.0
            waypoint = (unit['latitude'] + 0.2 * math.cos(rad), unit['longitude'] + 0.2 * math.sin(rad))
            return self.set_course(unit_guid, waypoint)
        else:
            return False

    def get_reference_points(self, side: str, name: str) -> List[Dict]:
        result = self._communicate(f"ScenEdit_GetReferencePoints({{side='{side}', area={{'{name}'}}}})")
        points = [p[1] for p in LuaParser(result).parse_array()]
        for p in points:
            p['latitude'] = float(p['latitude'].replace(",", "."))
            p['longitude'] = float(p['longitude'].replace(",", "."))
        return points

    def set_reference_point(self, side: Optional[str], name: Optional[str], latitude: float, longitude: float) -> Dict:
        msg = f"ScenEdit_SetReferencePoint({{side='{side}', name='{name}', lat={latitude}, lon={longitude}}})"
        result = self._communicate(msg)
        _, rp = LuaParser(result).parse_object()
        rp['latitude'] = float(rp['latitude'].replace(",", "."))
        rp['longitude'] = float(rp['longitude'].replace(",", "."))
        return rp

    def add_reference_point(self, side: str, name: str, latitude: float, longitude: float) -> str:
        msg = f"ScenEdit_AddReferencePoint({{side='{side}', name='{name}', lat={latitude}, lon={longitude}}})"
        result = self._communicate(msg)
        _, rp = LuaParser(result).parse_object()
        return rp['guid']

    def get_score(self, side: str) -> int:
        result = self._communicate(f"ScenEdit_GetScore('{side}')")
        return int(result)

    def set_trigger(self, mode: str, tr_type: str, name: str, target_filter: dict[str, str],
                    params: str = None) -> bool:
        msg = f"ScenEdit_SetTrigger({{mode='{mode}', type='{tr_type}', name='{name}'"
        if target_filter:
            msg += ", targetfilter=" + CMOConnector._dict_to_str(target_filter)
        if params:
            msg += ", " + params
        msg += "})"
        result = self._communicate(msg)
        return result != 'nil' and not result.startswith("Internal ERROR")

    def set_action(self, mode: str, ac_type: str, name: str, params: str) -> bool:
        msg = f"ScenEdit_SetAction({{mode='{mode}', type='{ac_type}', name='{name}', {params}}})"
        result = self._communicate(msg)
        return result != 'nil' and not result.startswith("Internal ERROR")

    def set_event(self, name: str, event_update: str):
        msg = f"ScenEdit_SetEvent('{name}', {{{event_update}}})"
        result = self._communicate(msg)
        return result != 'nil' and not result.startswith("Internal ERROR")

    def set_event_trigger(self, event_name: str, event_tca_update: str) -> bool:
        msg = f"ScenEdit_SetEventTrigger('{event_name}', {{{event_tca_update}}})"
        result = self._communicate(msg)
        return result != 'nil' and not result.startswith("Internal ERROR")

    def set_event_action(self, event_name: str, event_tca_update: str) -> bool:
        msg = f"ScenEdit_SetEventAction('{event_name}', {{{event_tca_update}}})"
        result = self._communicate(msg)
        return result != 'nil' and not result.startswith("Internal ERROR")

    def add_unit(self, unit_type: str, name: str, side: str, dbid: int, latitude: float, longitude: float,
                 heading: float, altitude: Optional[float] = None, loadoutid: Optional[int] = None) -> str:
        msg = f"ScenEdit_AddUnit({{type='{unit_type}', name='{name}', side='{side}', " \
              f"dbid={dbid}, lat={latitude}, lon={longitude}, heading={heading}" \
              f"{f', alt={altitude}' if altitude else ''}" \
              f"{f', loadoutid={loadoutid}' if loadoutid else ''}}})"
        result = self._communicate(msg)
        _, rp = LuaParser(result).parse_object()
        return rp['guid']

    def current_time(self) -> datetime:
        unix_timestamp = self._communicate("ScenEdit_CurrentTime()", ResponseType.SINGLE)
        return datetime.utcfromtimestamp(int(unix_timestamp))

    def save_scen(self, out_file=Union[Path, str]):
        escaped_out_file = str(out_file).replace('\\', '\\\\')
        res = self._communicate(f"Command_SaveScen(\"{escaped_out_file}\")", ResponseType.SINGLE)
        if res != "nil":
            raise CMOConnectorException("Command_SaveScen() failed")

    def export_scenario_to_xml(self) -> str:
        xml = self._communicate("ScenEdit_ExportScenarioToXML()", ResponseType.XML)
        return xml

    def import_scenario_from_xml(self, xml: str) -> ():
        """
            Warning: the 'py_helpers.lua' script must be loaded before calling this function
        """
        with open(self.xml_temp_file, mode="wb") as fp:
            fp.write(xml.encode(CMOConnector.MSG_ENCODING))
        result = self._communicate(f'LoadXMLScenario("{self.xml_temp_file}")', ResponseType.SINGLE)
        if result != "'Yes'":
            raise CMOConnectorException("import_scenario_from_xml() failed")

    def import_scenario_from_xml_file(self, file_name: str) -> None:
        """
            Load an XML scenario file previously saved with ScenEdit_ExportScenarioToXML()

            Warning: the 'py_helpers.lua' script must be loaded before calling this function
        :param file_name: the xml file to be imported
        """
        escaped_file_name = file_name.replace('\\', '\\\\')
        result = self._communicate(f'LoadXMLScenario("{escaped_file_name}")', ResponseType.SINGLE)
        if result != "'Yes'":
            raise CMOConnectorException("import_scenario_from_xml_file() failed")

    def run_period(self, hh_mm_ss: Tuple[int, int, int], real_time_mode=False) -> ():
        if real_time_mode:
            time.sleep(hh_mm_ss[0] * 3600 + hh_mm_ss[1] * 60 + hh_mm_ss[2])
        else:
            msg = f"VP_RunForTimeAndHalt({{Time='{hh_mm_ss[0]:02d}:{hh_mm_ss[1]:02d}:{hh_mm_ss[2]:02d}'}})"
            result = self._communicate(msg, ResponseType.SINGLE)
            if not result.startswith('OK'):
                raise CMOConnectorException(f"VP_RunForTimeAndHalt() returned {result}")
            while self.get_scenario()['Status'] == 'Running':
                time.sleep(CMOConnector.RUN_PERIOD_POLLING_INTERVAL)

    def set_time_compression(self, compression: int) -> None:
        self._communicate(f"VP_SetTimeCompression({compression})", ResponseType.SINGLE)

    def set_simulation_fidelity(self, fidelity: float) -> None:
        if fidelity not in [0.1, 1, 5]:
            raise RuntimeError("Fidelity argument must either be 0.1, 1 or 5")
        self._communicate(f"ScenEdit_SetSimulationFidelity({fidelity})", ResponseType.SINGLE)

    def get_contacts(self, side: str) -> List[Dict]:
        result = self._communicate(f"ScenEdit_GetContacts('{side}')")
        return [p[1] for p in LuaParser(result).parse_array()]

    def get_contact_attr(self, side: str, contact_guid: str, attribute: str, as_float=False) -> Union[str, float]:
        msg = f'ScenEdit_GetContact({{side="{side}", guid="{contact_guid}"}}).{attribute}\n'
        result = self._communicate(msg, ResponseType.SINGLE)
        return float(result.replace(",", ".")) if as_float else result

    def attack_contact(self, unit_guid: str, contact_id: str, mode: int, mount: Optional[int], weapon: int,
                       qty: int) -> bool:
        if mount:
            attack_options = f"{{mode='{mode}', mount={mount}, weapon={weapon}, qty={qty}}}"
        else:
            attack_options = f"{{mode='{mode}', weapon={weapon}, qty={qty}}}"
        result = self._communicate(f"ScenEdit_AttackContact('{unit_guid}', '{contact_id}', {attack_options})")
        return result == 'Yes'

    def run_script(self, script: str):
        """ Warning: this function doesn't return an error in case of failure!

        :param script: fully qualified script path
        """
        escaped_file_name = script.replace('\\', '\\\\')
        self._send(f'ScenEdit_RunScript("{escaped_file_name}\0", true)')

    @staticmethod
    def _parse_object(lua_str: str) -> Optional[dict]:
        if lua_str == 'nil':
            return None
        else:
            lap = LuaParser(lua_str)
            cls_name, obj = lap.parse_object()
            return obj

    @staticmethod
    def _dict_to_str(d: Union[dict, str], recursive: bool = False) -> str:
        if isinstance(d, dict):
            str_d = ",".join([f"{k}='{CMOConnector._dict_to_str(d[k], recursive) if recursive else d[k]}'" for k in d])
            return "{" + str_d + "}"
        else:
            return d
