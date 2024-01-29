// components/ExtractedFields.js
import { useState, useEffect, useRef } from "react";

const ExtractedFields = ({ doc_id, handleClick }) => {
  const [extractedData, setExtractedData] = useState(require("./dummy.json"));
  const [nodata, setNoData] = useState(false);
  const [isLoading, setLoading] = useState(true);
  const [changed, setChanged] = useState(false);
  const [selectedField, setSelectedField] = useState(null);

  const handleFieldChange = (fieldName, updatedValue) => {
    setExtractedData((prevValues) => {
      const newData = { ...prevValues };
      const fieldLevels = fieldName.split('.');
  
      let currentLevel = newData;
      for (let i = 0; i < fieldLevels.length - 1; i++) {
        const level = fieldLevels[i];
        currentLevel[level] = { ...(currentLevel[level] || {}) };
        currentLevel = currentLevel[level];
      }
  
      currentLevel[fieldLevels[fieldLevels.length - 1]].text = updatedValue;
  
      setChanged(true);
      setNoData(false);
  
      return newData;
    });
  };
  

  const handleSave = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:8000/db_connect/data_table/save_data/${doc_id}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(extractedData),
      });
  
      if (response.ok) {
        console.log('Data saved successfully');
        setChanged(false);
      } else {
        console.error('Failed to save data');
      }
    } catch (error) {
      console.error('Error saving data:', error);
    }
  };

  const handleDiscard = async () => {
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
          setNoData(true)
        }
        setLoading(false);
        setChanged(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setLoading(false);
      });
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
          setNoData(true)
        }
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setLoading(false);
      });
  }, [doc_id]);

  const renderField = (fieldName, fieldValue) => {
    if (fieldName?.toLowerCase() === "filename") {
      return null; // Do not render for fieldName "filename"
    }
    if (fieldName !== 'Table'){

      return (
        <div
          className={`m-1 cursor-default border p-2 rounded-md shadow-md transition duration-300 ease-in-out hover:shadow-lg ${
            selectedField === fieldName ? 'bg-red-200' : 'bg-white'
          }`}
          key={fieldName}
          onClick={(location) => {handleClick(fieldValue.location); setSelectedField(fieldName)}}
        >
          <p className="font-semibold mb-2 text-indigo-700">
            {fieldName}
          </p>
          <textarea
            className="text-gray-800 bg-blue-50 rounded-md border border-blue-300 p-2 focus:outline-none w-full "
            value={fieldValue.text}
            style={{
              minHeight: "40px",
              height: "auto",
              maxHeight: "200px",
            }}
            onChange={(e) => {
              handleFieldChange(fieldName, e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = e.target.scrollHeight + "px";
            }}
          />
        </div>
      );
    }
  };

  // console.log(extractedData)

  return (
    <div className="bg-white w-[30vw] text-center shadow-lg bg-gray-300">
      {isLoading && <p className="text-gray-500 p-1">Loading...</p>}
      {nodata && <p className="text-red-400 p-1">No Extracted Data Found</p>}
      {changed && (
        <p className="p-2 flex justify-center text-auto items-center space-x-4">
          <button onClick={handleSave} className="bg-green-600 hover:bg-green-800 text-white sm:px-2 md:px-2 lg:px-3 xl:px-4 sm:text-xs md:text-xs lg:text-lg xl:text-lg rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300">
            Save
          </button>
          <p className="">/</p>
          <button onClick={handleDiscard} className="bg-red-500 hover:bg-red-700 text-white sm:px-2 md:px-2 lg:px-3 xl:px-4 sm:text-xs md:text-xs lg:text-lg xl:text-lg rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300">
            Discard
          </button>
          <p className="text-xs">changes</p>
        </p>
      )}
      <div className="h-[86vh] overflow-y-auto shadow-xl sm:text-xs md:text-md lg:text-lg xl:text-xl">
        {Object.entries(extractedData).map(([fieldName, fieldValue]) =>
          renderField(fieldName, fieldValue)
        )}
      </div>
    </div>
  );
};

export default ExtractedFields;
