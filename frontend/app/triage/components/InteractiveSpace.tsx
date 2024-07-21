import { useState, useEffect } from "react";
import DocViewer from "@/app/triage/components/DocViewer";
import ExtractedFields from "@/app/triage/components/ExtractedFields";
import BACKEND_URLS from "@/app/BackendUrls";

interface InteractiveSpaceProps {
  doc_id: string | null;
  json_type: string | null;
}

const InteractiveSpace: React.FC<InteractiveSpaceProps> = ({
  doc_id,
  json_type,
}) => {
  const [boxLocation, setBoxLocation] = useState<Record<string, any> | null>(
    null
  );
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [selectedRow, setSelectedRow] = useState<number | null>(null);
  const [colName, setColName] = useState<string | null>(null);
  const [view, setView] = useState("General");

  const [extractedData, setExtractedData] = useState<{
    [key: string]: any;
  } | null>(null);

  const [noData, setNoData] = useState(false);
  const [isLoading, setLoading] = useState(true);
  const [dataChanged, setDataChanged] = useState(false);

  const handleSingleValuedFieldChange = (
    fieldName: string | null,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => {
    if (!fieldName) {
      return;
    }

    setExtractedData((prevValues: { [key: string]: any }) => {
      const newData = { ...prevValues };
      const fieldLevels = fieldName.split(".");
      let currentLevel = newData;
      for (let i = 0; i < fieldLevels.length - 1; i++) {
        const level = fieldLevels[i];
        currentLevel[level] = { ...(currentLevel[level] || {}) };
        currentLevel = currentLevel[level];
      }
      if (instruction === "update value") {
        currentLevel[fieldLevels[fieldLevels.length - 1]].text = value
          ? value
          : "";
      }
      if (instruction === "add bbox" || instruction === "del bbox") {
        currentLevel[fieldLevels[fieldLevels.length - 1]].location = location
          ? location
          : { pageNo: 0, ltwh: [0, 0, 0, 0] };

        setBoxLocation(location);
      }
      return newData;
    });
    setDataChanged(true);
  };

  const handleNestedFieldChange = (
    fieldName: string,
    index: number | null,
    colName: string | null,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => {
    if (index == null || !colName) {
      return;
    }
    // console.log(fieldName, index, colName, value, instruction)

    setExtractedData((prevData) => {
      const newData = { ...prevData };
      const updatedItem = { ...newData[fieldName][index]};
      if (instruction === "update value") {
        updatedItem[colName] = {
          ...updatedItem[colName],
          text: value ? value : "",
        };
      }
      if (instruction === "add bbox" || instruction === "del bbox") {
        updatedItem[colName] = {
          ...updatedItem[colName],
          location: location ? location : { pageNo: 0, ltwh: [0, 0, 0, 0] },
        };
        setBoxLocation(location);
      }
      newData[fieldName][index] = updatedItem;
      return newData;
    });
    setDataChanged(true);
  };

  const handleNestedRowDelete = (fieldName: string, index: number) => {
    setExtractedData((prevData) => {
      const newData = { ...prevData };
      const updatedTable = [...newData[fieldName]];
  
      if (updatedTable.length > 1) {
        updatedTable.splice(index, 1);
        newData[fieldName] = updatedTable;
      }
      else{
        handleNestedRowAdd(fieldName)
        handleNestedRowDelete(fieldName, index)
      }
  
      return newData;
    });
  
    setDataChanged(true);
  };
  

  const handleNestedRowAdd = (fieldName: string) => {
    setExtractedData((prevData) => {
      const newData = { ...prevData };
      const lastRowIndex = newData[fieldName].length - 1;

      const lastRow = newData[fieldName][lastRowIndex];

      const newRow = { ...lastRow };

      for (const field in lastRow) {
        newRow[field] = {
          text: "",
          location: {
            pageNo: 0,
            ltwh: [0, 0, 0, 0],
          },
        };
      }

      newData[fieldName] = [...newData[fieldName], { ...newRow }];

      return newData;
    });

    setDataChanged(true);
  };

  const handleSave = async () => {
    console.log(extractedData);
    try {
      const response = await fetch(
        `${BACKEND_URLS.uploadGTJsonUrl}/${doc_id}/`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(extractedData),
        }
      );

      if (response.ok) {
        console.log("Data saved successfully");
        setDataChanged(false);
        setNoData(false);
      } else {
        console.error("Failed to save data");
      }
    } catch (error) {
      console.error("Error saving data:", error);
    }
  };

  const handleDiscard = async () => {
    try {
      const response = await fetch(
        `${BACKEND_URLS.getJsonUrl}/${json_type}/${doc_id}`
      );
      const data = await response.json();
      if (data && !data.detail) {
        const initialData = data;
        setExtractedData(initialData);
        setDataChanged(false);
      } else {
        setExtractedData((prevData) => {
          return {};
        });
        setNoData(true);
        setDataChanged(false);
      }
      setLoading(false);
    } catch (error) {
      console.error("Error fetching data:", error);
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const response = await fetch(
          `${BACKEND_URLS.getJsonUrl}/${json_type}/${doc_id}`
        );
        const data = await response.json();
        if (data && !data.detail) {
          const initialData = data;
          setExtractedData(initialData);
          // setExtractedData(require("./tally.json"));
          setDataChanged(false);
        } else {
          setExtractedData(null);
          setNoData(true);
        }
        setLoading(false);
      } catch (error) {
        console.error("Error fetching data:", error);
        setLoading(false);
      }
    };

    fetchData();
  }, [doc_id]);

  const handleFieldClick = (
    fieldName: string,
    index: number | null,
    colName: string | null,
    location: Record<string, any>
  ) => {

    // console.log(index)
    // console.log(fieldName)
    // console.log(location)

    if (location.pageNo !== 0) {
      setSelectedRow(index);
      setBoxLocation(location);
      setSelectedField(fieldName);
      setColName(colName);
    } else {
      setSelectedRow(index);
      setBoxLocation(null);
      setSelectedField(fieldName);
      setColName(colName);
    }
  };

  const handleChangeView = (viewType: string) => {
    setView(viewType);
    setSelectedField(null)
  };

  return (
    <div className={`${view === "General" ? "flex" : ""}`}>
      <DocViewer
        doc_id={doc_id}
        boxLocation={boxLocation}
        viewType={view}
        handleSingleValuedFieldChange={handleSingleValuedFieldChange}
        handleNestedFieldChange={handleNestedFieldChange}
        selectedRow={selectedRow}
        selectedField={selectedField}
        colName={colName}
        dataChanged={dataChanged}
      />
      <ExtractedFields
        handleFieldClick={handleFieldClick}
        handleChangeView={handleChangeView}
        viewType={view}
        selectedField={selectedField}
        extractedData={extractedData}
        handleSingleValuedFieldChange={handleSingleValuedFieldChange}
        handleNestedFieldChange={handleNestedFieldChange}
        handleNestedRowDelete={handleNestedRowDelete}
        handleNestedRowAdd={handleNestedRowAdd}
        isLoading={isLoading}
        nodata={noData}
        dataChanged={dataChanged}
        handleSave={handleSave}
        handleDiscard={handleDiscard}
      />
    </div>
  );
};

export default InteractiveSpace;
