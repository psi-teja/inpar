import React, { useEffect, useState } from "react";
import AddField from "./AddField";

interface TableFieldsProps {
  fieldName: string;
  fieldValue: any;
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
  handleFieldClick: (
    fieldName: string,
    index: number | null,
    colName: string | null,
    location: Record<string, any>
  ) => void;
}

interface DisplayCols {
  [key: string]: boolean;
}

const TableFields: React.FC<TableFieldsProps> = ({
  fieldName,
  fieldValue,
  handleNestedFieldChange,
  handleNestedRowDelete,
  handleNestedRowAdd,
  handleFieldClick,
}) => {
  const [currIndex, setCurrIndex] = useState<number | null>(null);
  const [currField, setCurrField] = useState<string | null>(null);
  const [displayCols, setDisplayCols] = useState<DisplayCols>({});

  const changeCurr = (index: number, fieldName: string) => {
    setCurrIndex(index);
    setCurrField(fieldName);
  };

  const handleAddField = (fieldName: string) => {
    setDisplayCols((prevData) => {
      if (prevData === null) {
        return { [fieldName]: true };
      }
      const newData = { ...prevData, [fieldName]: !prevData[fieldName] };
      return newData;
    });
  };

  const handleSelectAll = (selectAll: boolean) => {
    const newDisplayCols = Object.keys(displayCols).reduce((acc, fieldName) => {
      acc[fieldName] = selectAll;
      return acc;
    }, {} as DisplayCols);
    setDisplayCols(newDisplayCols);
  };

  useEffect(() => {
    let predefinedFields: string[] = [];

    if (fieldName === "LedgerDetails") {
      predefinedFields = ['LedgerName', 'LedgerRate', 'LedgerAmount'];
    } else {
      predefinedFields = [
        "ItemBox",
        "ItemName",
        "ItemDescription",
        "HSNSACCode",
        "BilledQty",
        "ActualQty",
        "CGSTAmount",
        "DiscountAmount",
        "DiscountRate",
        "IGSTAmount",
        "ItemRate",
        "ItemRateUOM",
        "SGSTAmount",
        "ItemAmount",
      ];
    }

    const initialDisplayCols: DisplayCols = {};

    // Ensure all predefinedFields exist in every row, else add dummy value
    for (const field of predefinedFields) {
      fieldValue.forEach((row: any) => {
        if (!row[field]) {
          row[field] = {
            text: "",
            location: { pageNo: 0, ltwh: [0, 0, 0, 0, 0] }
          };
        }
      });

      initialDisplayCols[field] = fieldValue.some(
        (row: any) => row[field]?.text !== ""
      );
    }

    setDisplayCols(initialDisplayCols);
  }, [fieldName]);

  return (
    <div className="h-[25vh] overflow-auto">
      <table className="min-w-full bg-white">
        <thead className="sticky top-0 z-10">
          <tr className="">
            <th className="sticky left-0 bg-gray-100">
              <AddField
                displayCols={displayCols}
                handleAddField={handleAddField}
                handleSelectAll={handleSelectAll}
              />
            </th>
            {Object.entries(displayCols).map(
              ([fieldName, value]) =>
                value && (
                  <th
                    key={fieldName}
                    className={`px-2 text-left font-medium text-sm ${fieldName === currField ? "bg-cyan-300" : "bg-blue-500 text-white"}`}
                  >
                    {fieldName}
                  </th>
                )
            )}
          </tr>
        </thead>
        <tbody>
          {fieldValue.map((row: any, index: number) => (
            <tr
              key={index}
              className={`p-0 border-b ${index === currIndex ? "bg-gray-200" : ""}`}
            >
              <td
                className={`sticky left-0 ${index === currIndex ? "bg-cyan-300" : "bg-gray-100"}`}
              >
                <button
                  className={`px-3 text-xl font-bold rounded hover:bg-red-500 text-black focus:outline-none hover:text-white`}
                  onClick={(e) => handleNestedRowDelete(fieldName, index)}
                >
                  -
                </button>
              </td>
              {Object.entries(displayCols).map(
                ([colName, value]) =>
                  value && (
                    <td
                      key={colName}
                      onFocus={() => {
                        handleFieldClick(fieldName, index, colName, row[colName].location);
                        changeCurr(index, colName);
                      }}
                      onClick={() => {
                        handleFieldClick(fieldName, index, colName, row[colName].location);
                        changeCurr(index, colName);
                      }}
                      className={`p-1 ${colName === currField && index === currIndex ? "bg-red-200" : ""}`}
                    >
                      {colName !== "id" ? (
                        <div className="flex justify-content items-center">
                          <textarea
                            value={row[colName]?.text || ''}
                            className="p-2 m-1 h-10 rounded-md border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 overflow-auto resize"
                            onChange={(e) =>
                              handleNestedFieldChange(
                                fieldName,
                                index,
                                colName,
                                e.target.value,
                                row[colName]?.location,
                                "update value"
                              )
                            }
                          />

                          {row[colName]?.location?.pageNo !== 0 && (
                            <button
                              onClick={(e) =>
                                handleNestedFieldChange(
                                  fieldName,
                                  index,
                                  colName,
                                  row[colName].text,
                                  null,
                                  "del bbox"
                                )
                              }
                              disabled={colName !== currField || index !== currIndex}
                              className="relative"
                            >
                              <img
                                src="rect.png" // Replace with the actual path to your PNG image
                                alt="Draw Box"
                                className="h-4 w-5 m-2" // Adjust the height and width of the image as needed
                              />
                              {colName === currField && index === currIndex && (
                                <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center opacity-0 transition-opacity duration-300 ease-in-out hover:opacity-100">
                                  <div className="text-red-500">
                                    <svg
                                      className="h-7 w-7"
                                      xmlns="http://www.w3.org/2000/svg"
                                      fill="none"
                                      viewBox="0 0 24 24"
                                      stroke="currentColor"
                                      aria-hidden="true"
                                    >
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth="2"
                                        d="M6 18L18 6M6 6l12 12"
                                      />
                                    </svg>
                                  </div>
                                </div>
                              )}
                            </button>
                          )}
                        </div>
                      ) : (
                        row[colName].text
                      )}
                    </td>
                  )
              )}
            </tr>
          ))}
        </tbody>

      </table>
      <button
        className="p-2 m-2 font-bold text-black rounded hover:bg-gray-400 hover:text-white focus:outline-none"
        onClick={(e) => handleNestedRowAdd(fieldName)}
      >
        + Add Row
      </button>
    </div>
  );
};

export default TableFields;
