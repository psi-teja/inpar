import React, { useEffect, useState } from "react";
import AddField from "./AddField";

interface TableFieldsProps {
  fieldValue: any;
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
  handleFieldClick: (
    index: number|null,
    fieldName: string | null,
    boxLocation: Record<string, any>
  ) => void;
}

interface DisplayCols {
  [key: string]: boolean;
}

const TableFields: React.FC<TableFieldsProps> = ({
  fieldValue,
  handleNestedFieldChange,
  handleTableRowDelete,
  handleTableRowAdd,
  handleFieldClick,
}) => {
  const [currIndex, setCurrIndex] = useState(2);
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

  useEffect(() => {
    const initialDisplayCols: DisplayCols = {};

    for (const field in fieldValue[0]) {
      initialDisplayCols[field] = fieldValue.every(
        (row: any) => row[field].text === ""
      )
        ? false
        : true;
    }

    setDisplayCols(initialDisplayCols);
  }, [fieldValue]);

  return (
    <div className="h-[26vh] overflow-auto">
      <table className="min-w-full bg-white">
        <thead className="sticky top-0 z-10 bg-blue-300">
          <tr>
            <th className="sticky left-0 bg-blue-100 border-r border-b border-solid border-gray-400">
              <AddField
                displayCols={displayCols}
                handleAddField={handleAddField}
              />
            </th>
            {Object.entries(displayCols).map(
              ([fieldName, value]) =>
                value == true && (
                  <th
                    key={fieldName}
                    className={`px-2  text-left border-r border-b border-solid border-gray-400 font-medium text-sm text-gray-700 ${
                      fieldName == currField ? "bg-red-200" : ""
                    }`}
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
              className={`p-0  ${index === currIndex ? "bg-gray-200" : ""}`}
            >
              <td
                className={`sticky left-0 border-b border-r border-solid border-gray-400   ${
                  index === currIndex ? "bg-gray-200" : "bg-blue-100"
                }`}
              >
                <button
                  className="px-3 text-2xl font-bold text-red-400 rounded hover:bg-red-700 hover:text-white focus:outline-none"
                  onClick={(e) => handleTableRowDelete(index)}
                >
                  -
                </button>
              </td>
              {Object.entries(displayCols).map(
                ([fieldName, value]) =>
                  value == true && (
                    <td
                      key={fieldName}
                      onFocus={() => {
                        handleFieldClick(index, fieldName, row[fieldName].location);
                        changeCurr(index, fieldName);
                      }}
                      onClick={() => {
                        handleFieldClick(index, fieldName, row[fieldName].location);
                        changeCurr(index, fieldName);
                      }}
                      className={`p-0 ${(fieldName == currField && index == currIndex)? "bg-red-200": ""}`}
                    >
                      {fieldName !== "id" ? (
                        <div className="flex justify-content items-center border-r border-b border-solid border-gray-400">
                          <input
                            value={row[fieldName]?.text}
                            className="p-1 m-1 h-8 text-xs overflow-x-auto leading-4 border border-gray-300 rounded focus:outline-none focus:border-blue-500 hover:border-blue-400 overflow-x-auto"
                            onChange={(e) =>
                              handleNestedFieldChange(
                                "Table",
                                index,
                                fieldName,
                                e.target.value,
                                row[fieldName].location,
                                "update value"
                              )
                            }
                          />
                          {row[fieldName]?.location?.pageNo !== 0 && (
                            <button
                              onClick={(e) =>
                                handleNestedFieldChange(
                                  "Table",
                                  index,
                                  fieldName,
                                  row[fieldName].text,
                                  null,
                                  "del bbox"
                                )
                              }
                              disabled={fieldName !== currField || index !== currIndex}
                              className="relative"
                            >
                              <img
                                src="rect.png" // Replace with the actual path to your PNG image
                                alt="Draw Box"
                                className="h-4 w-5 m-2" // Adjust the height and width of the image as needed
                              />
                              {(fieldName === currField && index === currIndex) && (
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
                                        stroke-linecap="round"
                                        stroke-linejoin="round"
                                        stroke-width="2"
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
                        row[fieldName].text
                      )}
                    </td>
                  )
              )}
            </tr>
          ))}
          <tr className="">
            <button
              className="px-3 m-1 text-lg font-bold text-green-700 rounded hover:bg-green-700 hover:text-white focus:outline-none"
              onClick={handleTableRowAdd}
            >
              +
            </button>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default TableFields;
