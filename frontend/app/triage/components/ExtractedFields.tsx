// components/ExtractedFields.js
import React, { useState, useEffect } from "react";
import TableView from "./TableView";
import DocInfo from "./DocInfoView";


interface ExtractedFieldsProps {
  doc_id: string | null;
  handleFieldClick: (fieldName: string | null, boxLocation: Record<string, any>) => void;
  handleChangeView: (viewType: string) => void;
  viewType: string;
  selectedField: string | null;
}

const ExtractedFields: React.FC<ExtractedFieldsProps> = ({
  doc_id,
  handleFieldClick,
  handleChangeView,
  viewType,
  selectedField,
}) => {
  const [extractedData, setExtractedData] = useState<{ [key: string]: any } | null>(null);

  const [nodata, setNoData] = useState(false);
  const [isLoading, setLoading] = useState(true);
  const [changed, setChanged] = useState(false);

  const handleDocInfoFieldChange = (fieldName: string, updatedValue: string) => {
    setExtractedData((prevValues: { [key: string]: any }) => {
      const newData = { ...prevValues };
      const fieldLevels = fieldName.split(".");
      let currentLevel = newData;
      for (let i = 0; i < fieldLevels.length - 1; i++) {
        const level = fieldLevels[i];
        currentLevel[level] = { ...(currentLevel[level] || {}) };
        currentLevel = currentLevel[level];
      }
      currentLevel[fieldLevels[fieldLevels.length - 1]].text = updatedValue;
      return newData;
    });
    setChanged(true);
  };
  
  

  const handleTableFieldChange = (
    index: string | number,
    field: string | number,
    value: any
  ) => {
    setExtractedData((prevData) => {
      const newData = { ...prevData };
      const updatedItem = {
        ...newData.Table[index],
        [field]: { ...newData.Table[index][field], ["text"]: value },
      };
      newData.Table[index] = updatedItem;
      return newData;
    });
    setChanged(true);
  };

  const handleTableRowDelete = (index: number) => {
    setExtractedData((prevData) => {
      const newData = { ...prevData };
      const updatedTable = [...newData.Table];
      updatedTable.splice(index, 1);
      newData.Table = updatedTable;
      return newData;
    });
    setChanged(true);
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

    setChanged(true);
  };

  const handleSave = async () => {
    console.log(extractedData)
    try {
      const response = await fetch(`http://127.0.0.1:8000/db_connect/data_table/save_data/${doc_id}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(extractedData),
      });

      if (response.ok) {
        console.log("Data saved successfully");
        setChanged(false);
        setNoData(false)
      } else {
        console.error("Failed to save data");
      }
    } catch (error) {
      console.error("Error saving data:", error);
    }
  };

  const handleDiscard = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/db_connect/data_table/get_data/${doc_id}`);
      const data = await response.json();
      if (data && !data.detail) {
        const initialData = data
        setExtractedData(initialData);
        setChanged(false);
      } else {
        setExtractedData((prevData) => {
          return require("./dummy.json");
        });
        setNoData(true);
        setChanged(false);
      }
      setLoading(false);
    } catch (error) {
      console.error("Error fetching data:", error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch(`http://127.0.0.1:8000/db_connect/data_table/get_data/${doc_id}`)
      .then((res) => res.json())
      .then((data) => {
        // console.log(data)
        if (data && data.doc_json_gt) {
          setExtractedData(data.doc_json_gt);
        }
        else if (data && data.doc_json_ai) {
          setExtractedData(data.doc_json_ai);
        }
        else{
          setExtractedData((prevData) => {
            return require("./dummy.json");
          })
          setNoData(true)
        }
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setLoading(false);
      });
  }, [doc_id]);

  const renderField = (fieldName: string, fieldValue: any) => {
    if (fieldName?.toLowerCase() === "filename") {
      return null; // Do not render for fieldName "filename"
    }
    if (fieldName !== "Table" && viewType === "DocInfo") {
      return (
        <DocInfo
          fieldName={fieldName}
          fieldValue={fieldValue}
          selectedField={selectedField}
          handleFieldClick={handleFieldClick}
          handleDocInfoFieldChange={handleDocInfoFieldChange}
        />
      );
    }
    if (fieldName === "Table" && viewType === "Table") {
      return (
        <div className="text-xs">
          <TableView
            fieldValue={fieldValue}
            handleChange={handleTableFieldChange}
            handleRowDelete={handleTableRowDelete}
            handleRowAdd={handleTableRowAdd}
            handleFieldClick={handleFieldClick}
          />
        </div>
      );
    }
  };

  return (
    <div
      className={`bg-white bg-opacity-0 ${
        viewType === "DocInfo" ? "w-[30vw]" : "mt-2"
      } text-center font-mono`}
    >
      {isLoading && <p className="text-gray-500 p-1">Loading...</p>}
      {nodata && <div>
        <p className="text-red-600">No Data Extracted</p>
        <p className="text-blue-600">feel free to add correct data and save</p>
        </div>}
      <div
        className={`${
          viewType == "DocInfo"
            ? "sm:mt-2 md:mt-3 lg:mt-4 xl:mt-4 order-last"
            : ""
        }`}
      >
        <div className="flex sm:text-xs md:text-xs lg:text-lg xl:text-lg ml-auto">
          <button
            className={`${
              viewType === "DocInfo"
                ? "bg-gradient-to-b from-blue-600 to-blue-100 border border-blue-500"
                : "bg-gray-300 text-gray-400 hover:bg-gradient-to-b from-blue-400 to-blue-100 hover:text-gray-600"
            } text-black p-1 ml-1 rounded-t`}
            onClick={() => handleChangeView("DocInfo")}
          >
            Doc Info
          </button>
          <button
            className={`${
              viewType === "Table"
                ? "bg-gradient-to-b from-blue-600 to-blue-100 border border-blue-500"
                : "bg-gray-300 text-gray-400 hover:bg-gradient-to-b from-blue-400 to-blue-100 hover:text-gray-600"
            } text-black p-1 mr-1 rounded-t`}
            onClick={() => handleChangeView("Table")}
          >
            Table
          </button>
        </div>
      </div>
      <div
        className={`${
          viewType === "DocInfo" ? "h-[74vh] overflow-y-auto" : ""
        }  border border-blue-400 shadow`}
      >
        {extractedData &&
          Object.entries(extractedData).map(([fieldName, fieldValue]) =>
            renderField(fieldName, fieldValue)
          )}
      </div>
      <div className="p-4 sm:mt-2 md:mt-3 lg:mt-4 xl:mt-4">
        <button
          onClick={handleSave}
          disabled={!changed}
          className={`${
            changed
              ? "bg-green-600 hover:bg-green-800"
              : "bg-gray-300 text-gray-400 cursor-not-allowed"
          } text-white p-1 mr-2 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300`}
        >
          Save
        </button>
        <button
          onClick={handleDiscard}
          disabled={!changed}
          className={`${
            changed
              ? "bg-red-500 hover:bg-red-700"
              : "bg-gray-300 text-gray-400 cursor-not-allowed"
          } text-white p-1 ml-2 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300`}
        >
          Discard
        </button>
      </div>
    </div>
  );
};

export default ExtractedFields;
