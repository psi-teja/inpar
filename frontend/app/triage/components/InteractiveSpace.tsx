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
  const [view, setView] = useState("DocInfo");

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
    fieldType: string,
    index: number | null,
    field: string | null,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => {
    if (index == null || !field) {
      return;
    }
    console.log(boxLocation)

    setExtractedData((prevData) => {
      const newData = { ...prevData };
      const updatedItem = { ...newData[fieldType][index] };
      if (instruction === "update value") {
        updatedItem[field] = {
          ...updatedItem[field],
          text: value ? value : "",
        };
      }
      if (instruction === "add bbox" || instruction === "del bbox") {
        updatedItem[field] = {
          ...updatedItem[field],
          location: location ? location : { pageNo: 0, ltwh: [0, 0, 0, 0] },
        };
        setBoxLocation(location);
      }
      newData.Table[index] = updatedItem;
      return newData;
    });
    setDataChanged(true);
  };

  const handleTableRowDelete = (index: number) => {
    setExtractedData((prevData) => {
      const newData = { ...prevData };
      const updatedTable = [...newData.Table];
      updatedTable.splice(index, 1);
      newData.Table = updatedTable;
      return newData;
    });
    setDataChanged(true);
  };

  const handleTableRowAdd = () => {
    setExtractedData((prevData) => {
      const newData = { ...prevData };
      const lastRowIndex = newData.Table.length - 1;

      const lastRow = newData.Table[lastRowIndex];

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

      newData.Table = [...newData.Table, { ...newRow }];

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
          setExtractedData({});
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

  const changeBox = (
    index: number | null,
    fieldName: string | null,
    location: Record<string, any>
  ) => {

    if (location.pageNo !== 0) {
      setSelectedRow(index);
      setBoxLocation(location);
      setSelectedField(fieldName);
    } else {
      setSelectedRow(index);
      setBoxLocation(null);
      setSelectedField(fieldName);
    }
  };

  const changeView = (viewType: string) => {
    setView(viewType);
  };

  return (
    <div className={`${view === "DocInfo" ? "flex" : ""}`}>
      <DocViewer
        doc_id={doc_id}
        boxLocation={boxLocation}
        viewType={view}
        handleSingleValuedFieldChange={handleSingleValuedFieldChange}
        handleNestedFieldChange={handleNestedFieldChange}
        selectedRow={selectedRow}
        selectedField={selectedField}
        dataChanged={dataChanged}
      />
      <ExtractedFields
        handleFieldClick={changeBox}
        handleChangeView={changeView}
        viewType={view}
        selectedField={selectedField}
        extractedData={extractedData}
        handleSingleValuedFieldChange={handleSingleValuedFieldChange}
        handleNestedFieldChange={handleNestedFieldChange}
        handleTableRowDelete={handleTableRowDelete}
        handleTableRowAdd={handleTableRowAdd}
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
