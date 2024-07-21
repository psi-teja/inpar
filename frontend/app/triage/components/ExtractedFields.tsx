// components/ExtractedFields.js
import React, { useState, useEffect } from "react";
import TableFields from "./TableFields";
import SingleValuedField from "./SingleValuedField";
import ToggleView from "./ToggleView";
import { FaExclamationCircle } from "react-icons/fa"; // Import icon from react-icons library
import AddField from "./AddField";

interface ExtractedFieldsProps {
  handleFieldClick: (
    fieldName: string,
    index: number | null,
    colName: string | null,
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
  handleNestedRowDelete: (fieldName: string, index: number) => void;
  handleNestedRowAdd: (fieldName: string) => void;
  isLoading: boolean;
  nodata: boolean;
  dataChanged: boolean;
  handleSave: () => void;
  handleDiscard: () => void;
}

interface DisplayFields {
  [key: string]: boolean;
}

const ExtractedFields: React.FC<ExtractedFieldsProps> = ({
  handleFieldClick,
  handleChangeView,
  viewType,
  selectedField,
  extractedData,
  handleSingleValuedFieldChange,
  handleNestedFieldChange,
  handleNestedRowDelete,
  handleNestedRowAdd,
  isLoading,
  nodata,
  dataChanged,
  handleSave,
  handleDiscard,
}) => {


  // Function to check if an object is empty
  const isEmptyObject = (obj: any) => Object.keys(obj).length === 0 && obj.constructor === Object;

  if (extractedData == null) {
    return (
      <div className="flex flex-col items-center justify-center h-screen w-[30vw]">
        <FaExclamationCircle className="text-5xl text-gray-400" /> {/* Icon */}
        <p className="text-xl text-gray-600 m-4">No data available</p> {/* Text */}
        <button
          onClick={() => window.location.reload()} // Refresh the page on button click
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Refresh
        </button>
      </div>
    );
  } else if (isEmptyObject(extractedData)) {
    return (
      <div className="flex flex-col items-center justify-center h-screen w-[30vw]">
        <FaExclamationCircle className="text-5xl text-gray-400 mb-4" /> {/* Icon */}
        <p className="text-xl text-gray-600">AI failed to extract any field</p> {/* Text */}
      </div>
    );
  }

  const [displayFields, setDisplayFields] = useState<DisplayFields>({});

  useEffect(() => {
    const predefinedFields: string[] = [
      "InvoiceDate",
      "InvoiceNumber",
      "BuyerAddress",
      "BuyerContactNo",
      "BuyerEmail",
      "BuyerGSTIN",
      "BuyerName",
      "BuyerOrderDate",
      "BuyerPAN",
      "BuyerState",
      "ConsigneeAddress",
      "ConsigneeContactNo",
      "ConsigneeEmail",
      "ConsigneeGSTIN",
      "ConsigneeName",
      "ConsigneePAN",
      "ConsigneeState",
      "Destination",
      "DispatchThrough",
      "DocumentType",
      "OrderNumber",
      "OtherReference",
      "PortofLoading",
      "ReferenceNumber",
      "SubAmount",
      "SupplierAddress",
      "SupplierContactNo",
      "SupplierEmail",
      "SupplierGSTIN",
      "SupplierName",
      "SupplierPAN",
      "SupplierState",
      "TermsofPayment",
      "TotalAmount",
      "Table",
      "LedgerDetails"];
    const initialDisplayFields: DisplayFields = {};

    predefinedFields.forEach(field => {
      if (!extractedData[field]) {
        if (field == "Table" || field == "LedgerDetails") {
          extractedData[field] = [{}]
        }
        else {
          extractedData[field] = {
            text: "",
            location: { pageNo: 0, ltwh: [0, 0, 0, 0] }
          };

        }
      }

      if (field !== "Table" && field !== "LedgerDetails") { initialDisplayFields[field] = (extractedData[field].text !== "" || extractedData[field].location.pageNo != 0); }
    });

    setDisplayFields(initialDisplayFields);
  }, []);

  const handleAddField = (fieldName: string) => {
    setDisplayFields(prevData => ({
      ...prevData,
      [fieldName]: !prevData[fieldName]
    }));
  };

  const handleSelectAll = (selectAll: boolean) => {
    const newDisplayCols = Object.keys(displayFields).reduce((acc, fieldName) => {
      acc[fieldName] = selectAll;
      return acc;
    }, {} as DisplayFields);
    setDisplayFields(newDisplayCols);
  };

  const renderField = (fieldName: string, fieldValue: any) => {
    if (fieldName?.toLowerCase() === "filename") {
      return null;
    }
    if (fieldName !== "Table" && viewType === "General" && fieldName !== "LedgerDetails" && displayFields[fieldName]) {
      return (
        <SingleValuedField
          fieldName={fieldName}
          fieldValue={fieldValue}
          selectedField={selectedField}
          handleFieldClick={handleFieldClick}
          handleSingleValuedFieldChange={handleSingleValuedFieldChange}
        />
      );
    }
    if (fieldName === "Table" && viewType === "Items") {
      return (
        <div className="text-xs">
          <TableFields
            fieldName={fieldName}
            fieldValue={fieldValue}
            handleNestedFieldChange={handleNestedFieldChange}
            handleNestedRowDelete={handleNestedRowDelete}
            handleNestedRowAdd={handleNestedRowAdd}
            handleFieldClick={handleFieldClick}
          />
        </div>
      );
    }
    if (fieldName === "LedgerDetails" && viewType === "Ledgers") {
      return (
        <div className="text-xs">
          <TableFields
            fieldName={fieldName}
            fieldValue={fieldValue}
            handleNestedFieldChange={handleNestedFieldChange}
            handleNestedRowDelete={handleNestedRowDelete}
            handleNestedRowAdd={handleNestedRowAdd}
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
      className={`bg-white bg-opacity-0 ${viewType === "General" ? "w-[30vw]" : "mt-2"
        } text-center font-mono`}
    >
      {isLoading && <p className="text-gray-500 p-1">Loading...</p>}
      {nodata && (
        <div>
          <p className="text-red-600">No Data Extracted</p>
        </div>
      )}
      <div
        className={`${viewType == "General"
          ? "sm:mt-2 md:mt-3 lg:mt-4 xl:mt-4 order-last"
          : ""
          }`}
      >
        <div className="flex justify-between sm:text-xs md:text-xs lg:text-lg xl:text-lg ml-auto">
          {viewType == "General" && <AddField displayCols={displayFields} handleAddField={handleAddField} handleSelectAll={handleSelectAll} />
          }
          <ToggleView viewType={viewType} handleChangeView={handleChangeView} />

          <div
            className="hover:bg-gray-200 rounded mr-2"
            title="Download JSON"
            onClick={() => downloadJSON(extractedData, "extractedData.json")}
          >
            <svg
              className="h-5 w-5 mt-2 text-black "
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
        className={`${viewType === "General" ? "h-[80vh] overflow-y-auto" : ""
          }  border shadow-inner`}
      >
        {extractedData &&
          Object.entries(extractedData).map(([fieldName, fieldValue]) =>
            renderField(fieldName, fieldValue)
          )}
      </div>
      <div className="p-4  flex justify-center space-x-4">
        <button
          onClick={handleSave}
          disabled={!dataChanged}
          className={`${dataChanged
            ? "bg-green-600 hover:bg-green-800"
            : "bg-gray-300 text-gray-400 cursor-not-allowed"
            } text-white py-2 px-4 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-green-300`}
        >
          Save
        </button>
        <button
          onClick={handleDiscard}
          disabled={!dataChanged}
          className={`${dataChanged
            ? "bg-red-500 hover:bg-red-700"
            : "bg-gray-300 text-gray-400 cursor-not-allowed"
            } text-white py-2 px-4 rounded-md transition duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-red-300`}
        >
          Discard
        </button>
      </div>

    </div>
  );
};

export default ExtractedFields;
