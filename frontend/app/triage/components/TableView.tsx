import React, { useEffect, useState } from "react";
import AddField from "./AddField";

interface TableViewProps {
  fieldValue: any;
  handleChange: (index: number, field: string, value: any) => void;
  handleRowDelete: (index: number) => void;
  handleRowAdd: () => void;
  handleFieldClick: (fieldName: string | null, boxLocation: Record<string, any>) => void;
}

interface DisplayCols {
  [key: string]: boolean;
}

const TableView: React.FC<TableViewProps> = ({
  fieldValue,
  handleChange,
  handleRowDelete,
  handleRowAdd,
  handleFieldClick,
}) => {
  const [currIndex, setCurrIndex] = useState(2);
  const [currField, setCurrField] = useState<string | null>(null);
  const [displayCols, setDisplayCols] = useState<DisplayCols>({});

  const changeCurr = (index: number, fieldName: string) => {
    setCurrIndex(index);
    setCurrField(fieldName)
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
            <th>
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
                    className={`p-1 border border-b-0 font-medium text-sm text-gray-700 ${(fieldName == currField)? "bg-red-200":""}`}
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
              className={`border border-b ${
                index === currIndex ? "bg-gray-200" : ""
              }`}
            >
              <td className="sticky left-0 border border-b-0">
                <button
                  className="px-2 text-2xl font-bold text-red-400 rounded hover:bg-red-700 hover:text-white focus:outline-none"
                  onClick={(e) => handleRowDelete(index)}
                >
                  -
                </button>
              </td>
              {Object.entries(displayCols).map(
                ([fieldName, value]) =>
                  value == true && (
                    <td
                      key={fieldName}
                      onClick={() => {
                        handleFieldClick(null, row[fieldName].location);
                        changeCurr(index, fieldName);
                      }}
                      
                    >
                      {fieldName !== "id" ? (
                        <input
                          value={row[fieldName].text}
                          className="p-1 m-1 h-8 text-xs overflow-x-auto leading-4 border border-gray-300 rounded focus:outline-none focus:border-blue-500 hover:border-blue-400 overflow-x-auto"
                          onChange={(e) =>
                            handleChange(index, fieldName, e.target.value)
                          }
                        />
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
              className="px-2 m-1 text-lg font-bold text-green-700 rounded hover:bg-green-700 hover:text-white focus:outline-none"
              onClick={handleRowAdd}
            >
              +
            </button>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default TableView;
