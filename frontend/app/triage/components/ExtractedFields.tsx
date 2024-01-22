// components/ExtractedFields.js
import { useState, useEffect } from "react";

const ExtractedFields = ({ doc_id }) => {
  const [extractedData, setExtractedData] = useState(require("./dummy.json"));
  const [nodata, setNoData] = useState(false);
  const [isLoading, setLoading] = useState(true);
  const [changed, setChanged] = useState(false);

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
      const response = await fetch(`http://127.0.0.1:8000/db_connect/save_data/${doc_id}`, {
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

  useEffect(() => {
    fetch(`http://127.0.0.1:8000/db_connect/data_table/${doc_id}`)
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
  }, []);

  const renderField = (fieldName, fieldValue) => {
    if (fieldName?.toLowerCase() === "filename") {
      return null; // Do not render for fieldName "filename"
    }
    if (!Array.isArray(fieldValue)){

      return (
        <div
          className="mb-4 border p-2 rounded-md shadow-md transition duration-300 ease-in-out hover:shadow-lg bg-white"
          key={fieldName}
        >
          <p className="text-lg font-semibold mb-2 text-indigo-700">
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
      {/* <h2 className="text-2xl font-bold text-gray-100 bg-gradient-to-r from-blue-500 to-blue-700 text-white py-3 rounded-md mb-4">
        Extracted Fields
      </h2> */}
      {isLoading && <p className="text-gray-500 p-1">Loading...</p>}
      {nodata && <p className="text-red-400 p-1">No Extracted Data Found</p>}
      {changed && (
        <p className="p-2 flex justify-center space-x-4">
          <button onClick={handleSave} className="bg-green-600 hover:bg-green-800 text-white py-1 px-6 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300">
            Save
          </button>
          <button className="bg-red-500 hover:bg-red-700 text-white py-1 px-6 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300">
            Discard
          </button>
        </p>
      )}
      <div className="h-[86vh] overflow-y-auto shadow-xl ">
        {Object.entries(extractedData).map(([fieldName, fieldValue]) =>
          renderField(fieldName, fieldValue)
        )}
      </div>
    </div>
  );
};

export default ExtractedFields;
