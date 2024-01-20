// components/ExtractedFields.js
"use client";
import { useState, useEffect } from "react";

const ExtractedFields = ({ doc_id }) => {
  const [data, setData] = useState(require("./dummy.json"));
  const [nodata, setNoData] = useState(false);
  const [isLoading, setLoading] = useState(true);
  const [changed, setChanged] = useState(false);

  const handleFieldChange = (fieldName, updatedValue) => {
    setData((prevValues) => ({
      ...prevValues,
      [fieldName]: updatedValue,
    }));
    setChanged(true);
    setNoData(false);
  };

  useEffect(() => {
    fetch(`http://127.0.0.1:8000/db_connect/data_table/${doc_id}/`)
      .then((res) => res.json())
      .then((data) => {
        setData(data.data.doc_json_ai);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
        setLoading(false);
        setNoData(true);
      });
  }, []);

  const renderField = (fieldName, fieldValue) => {
    if (fieldName?.toLowerCase() === "filename") {
      return null; // Do not render for fieldName "filename"
    }

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
          value={fieldValue}
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
  };

  return (
    <div className="bg-white w-[30vw] text-center shadow-lg bg-gray-300">
      {/* <h2 className="text-2xl font-bold text-gray-100 bg-gradient-to-r from-blue-500 to-blue-700 text-white py-3 rounded-md mb-4">
        Extracted Fields
      </h2> */}
      {isLoading && <p className="text-gray-500 p-1">Loading...</p>}
      {nodata && <p className="text-red-400 p-1">No Extracted Data Found</p>}
      {changed && (
        <p className="p-2 flex justify-center space-x-4">
          <button className="bg-green-600 hover:bg-green-800 text-white py-1 px-6 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300">
            Save
          </button>
          <button className="bg-red-500 hover:bg-red-700 text-white py-1 px-6 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300">
            Discard
          </button>
        </p>
      )}
      <div className="h-[86vh] overflow-y-auto shadow-xl ">
        {Object.entries(data).map(([fieldName, fieldValue]) =>
          renderField(fieldName, fieldValue[0]?.text)
        )}
      </div>
    </div>
  );
};

export default ExtractedFields;
