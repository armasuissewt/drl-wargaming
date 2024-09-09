-- Load a scenario file previously saved with ScenEdit_ExportScenarioToXML()
function LoadXMLScenario(file_name)
    filehandle=io.open(file_name, "r")
    content = filehandle:read("*a")
    filehandle:close()
    return ScenEdit_ImportScenarioFromXML({XML=content})
end

function TestSum(a, b)
    return a + b
end
