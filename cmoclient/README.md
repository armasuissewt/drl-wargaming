# CmoClient

A python client for the Command Modern Operations LUA interface.

## Example

Start a CMO instance with a command similar to the following:

```powershell
& "C:\Program Files (x86)\Command Professional Edition 2\CommandCLI.exe" -mode I -scenfile "SomeScenario.scen"
```

Then run the following:

```python
from cmoclient.cmo_connector import CMOConnector

conn = CMOConnector()
conn.connect()

print(conn.get_scenario())
# ... other calls using the conn object

conn.disconnect()
```

## Citation

Please cite this code as specified in the [README.md](https://github.com/armasuissewt/drl-wargaming/blob/master/README.md) file in the root of this repository.

## License

MIT License

Copyright (c) 2024 IDSIA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.