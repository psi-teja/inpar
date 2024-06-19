import React, { useEffect, useState } from "react";
import AddField from "./AddField";

interface LedgerDetailsFieldsProps {
  fieldName: string;
  fieldValue: Record<string, any>;
  selectedField: string | null;
  handleFieldClick: (
    index: number | null,
    fieldName: string,
    boxLocation: Record<string, any>
  ) => void;
  handleNestedFieldChange: (
    fieldType: string,
    index: number | null,
    field: string | null,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => void;
}

const LedgerDetailsFields: React.FC<LedgerDetailsFieldsProps> = ({
  fieldName,
  fieldValue,
  handleFieldClick,
  handleNestedFieldChange,
}) => {

  console.log(fieldValue)

  return (
    <div className="m-2 border border-gray-400 sm:text-xs md:text-xs lg:text-lg xl:text-lg cursor-default p-2 rounded-md shadow-md transition duration-300 ease-in-out hover:shadow-lg">
      <p className="font-semibold m-1 text-indigo-700 text-center">
        {fieldName}
      </p>
      <table className="table">
        <thead>
          <tr className="text-indigo-700">
            <th>Name</th>
            <th>Rate</th>
            <th>Amount</th>
          </tr>
        </thead>
        <tbody>
          {fieldValue.map((ledger: any, index: number) => (
            <tr key={index}>
              {Object.entries(ledger).map(
                (propertyName: any, propertyValue: any) => (
                  <td key={ledger}>
                    <input
                      className="text-gray-800 bg-blue-50 rounded-md border border-blue-300 p-2 focus:outline-none w-full"
                      type="text"
                      value={propertyValue.text}
                      onChange={(e) => {
                        console.log(propertyName, propertyValue);
                        handleNestedFieldChange(
                          fieldName,
                          index,
                          propertyName,
                          e.target.value,
                          propertyValue.location,
                          "update value"
                        );
                        handleFieldClick(
                          index,
                          fieldName,
                          propertyValue.location
                        );
                      }}
                    />
                  </td>
                )
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default LedgerDetailsFields;
