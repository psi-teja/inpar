import { useState } from "react";
import React from 'react';

interface DisplayCols {
  [key: string]: boolean;
}

interface AddFieldProps {
  displayCols: DisplayCols;
  handleAddField: (fieldName: string) => void;
}

const AddField: React.FC<AddFieldProps> = ({ displayCols, handleAddField }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div>
      <button
        className={`px-2 py-1 m-1 rounded ${
          isOpen ? "bg-red-500" : "bg-blue-500"
        }`}
        onClick={toggleDropdown}
      >
        Add
      </button>
      {isOpen && (
        <div className="relative">
          <div
            className="absolute top-[1vh] left-[4vw] rounded-xl h-[20vh] overflow-y-hidden shadow-xl p-4 border border-solid border-gray-600 bg-blue-200"
            style={{ overflowY: isOpen ? "auto" : "hidden" }}
            onMouseEnter={() => setIsOpen(true)}
            onMouseLeave={() => setIsOpen(false)}
          >
            {Object.entries(displayCols).map(([fieldName, b]) => (
              <div key={fieldName} className="flex m-1 text-md">
                <input
                  type="checkbox"
                  checked={b === true}
                  onChange={() => handleAddField(fieldName)}
                  className="sm:w-3 md:w-4 lg:w-5 xl:w-6 text-blue-300 ml-2"
                />
                <div
                  className={`ml-3 sm:text-xs md:text-md lg:text-lg xl:text-xl ${
                    b ? "text-blue-800" : "text-gray-500"
                  }`}
                >
                  {fieldName}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default AddField;
