// components/ExtractedFields.js
import React from "react";
import TableFields from "./TableFields";
import SingleValuedField from "./SingleValuedField";
import LedgerDetailsFields from "./LedgerDetailsFields";

interface ExtractedFieldsProps {
  handleFieldClick: (
    index: number | null,
    fieldName: string | null,
    location: Record<string, any>
  ) => void;
  handleChangeView: (viewType: string) => void;
  viewType: string;
  selectedField: string | null;
  extractedData: { [key: string]: any } | null;
  handleSingleValuedFieldChange: (
    fieldName: string,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => void;
  handleNestedFieldChange: (
    fieldType: string,
    index: number | null,
    field: string | null,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => void;
  handleTableRowDelete: (index: number) => void;
  handleTableRowAdd: () => void;
  isLoading: boolean;
  nodata: boolean;
  dataChanged: boolean;
  handleSave: () => void;
  handleDiscard: () => void;
}

const ExtractedFields: React.FC<ExtractedFieldsProps> = ({
  handleFieldClick,
  handleChangeView,
  viewType,
  selectedField,
  extractedData,
  handleSingleValuedFieldChange,
  handleNestedFieldChange,
  handleTableRowDelete,
  handleTableRowAdd,
  isLoading,
  nodata,
  dataChanged,
  handleSave,
  handleDiscard,
}) => {
  const renderField = (fieldName: string, fieldValue: any) => {
    if (fieldName?.toLowerCase() === "filename") {
      return null;
    }
    if (fieldName !== "Table" && viewType === "DocInfo") {
      if (fieldName !== "LedgerDetails") {
        return (
          <SingleValuedField
            fieldName={fieldName}
            fieldValue={fieldValue}
            selectedField={selectedField}
            handleFieldClick={handleFieldClick}
            handleSingleValuedFieldChange={handleSingleValuedFieldChange}
          />
        );
      } else {
        return (
            <LedgerDetailsFields
              fieldName={fieldName}
              fieldValue={fieldValue}
              selectedField={selectedField}
              handleFieldClick={handleFieldClick}
              handleNestedFieldChange={handleNestedFieldChange}
            />
        );
      }
    }
    if (fieldName === "Table" && viewType === "Table") {
      return (
        <div className="text-xs">
          <TableFields
            fieldValue={fieldValue}
            handleNestedFieldChange={handleNestedFieldChange}
            handleTableRowDelete={handleTableRowDelete}
            handleTableRowAdd={handleTableRowAdd}
            handleFieldClick={handleFieldClick}
          />
        </div>
      );
    }
  };

  const downloadJSON = (
    data: { [key: string]: any } | null,
    filename: string
  ) => {
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  };
  return (
    <div
      className={`bg-white bg-opacity-0 ${
        viewType === "DocInfo" ? "w-[30vw]" : "mt-2"
      } text-center font-mono`}
    >
      {isLoading && <p className="text-gray-500 p-1">Loading...</p>}
      {nodata && (
        <div>
          <p className="text-red-600">No Data Extracted</p>
        </div>
      )}
      <div
        className={`${
          viewType == "DocInfo"
            ? "sm:mt-2 md:mt-3 lg:mt-4 xl:mt-4 order-last"
            : ""
        }`}
      >
        <div className="flex justify-between sm:text-xs md:text-xs lg:text-lg xl:text-lg ml-auto">
          <div>
            <button
              className={`${
                viewType === "DocInfo"
                  ? "bg-gradient-to-b from-blue-300 to-blue-100"
                  : "bg-gray-300 text-gray-400 hover:bg-gradient-to-b from-blue-300 to-blue-100 hover:text-gray-500"
              } text-black p-1 ml-1 rounded-t`}
              onClick={() => handleChangeView("DocInfo")}
            >
              Doc Info
            </button>
            <button
              className={`${
                viewType === "Table"
                  ? "bg-gradient-to-b from-blue-300 to-blue-100 border border-blue-100"
                  : "bg-gray-300 text-gray-400 hover:bg-gradient-to-b from-blue-300 to-blue-100 hover:text-gray-600"
              } text-black p-1 mr-1 rounded-t`}
              onClick={() => handleChangeView("Table")}
            >
              Table
            </button>
          </div>
          <div
            className="hover:bg-gray-200 rounded mr-4"
            title="Download JSON"
            onClick={() => downloadJSON(extractedData, "extractedData.json")}
          >
            <svg
              className="h-5 w-5 text-black "
              width="24"
              height="24"
              viewBox="0 0 24 24"
              stroke-width="2"
              stroke="currentColor"
              fill="none"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              {" "}
              <path stroke="none" d="M0 0h24v24H0z" />{" "}
              <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2" />{" "}
              <polyline points="7 11 12 16 17 11" />{" "}
              <line x1="12" y1="4" x2="12" y2="16" />
            </svg>
          </div>
        </div>
      </div>
      <div
        className={`${
          viewType === "DocInfo" ? "h-[74vh] overflow-y-auto" : ""
        }  border border-blue-100 shadow`}
      >
        {extractedData &&
          Object.entries(extractedData).map(([fieldName, fieldValue]) =>
            renderField(fieldName, fieldValue)
          )}
      </div>
      <div className="p-4 sm:mt-2 md:mt-3 lg:mt-4 xl:mt-4">
        <button
          onClick={handleSave}
          disabled={!dataChanged}
          className={`${
            dataChanged
              ? "bg-green-600 hover:bg-green-800"
              : "bg-gray-300 text-gray-400 cursor-not-allowed"
          } text-white p-1 mr-2 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring focus:border-blue-300`}
        >
          Save
        </button>
        <button
          onClick={handleDiscard}
          disabled={!dataChanged}
          className={`${
            dataChanged
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
