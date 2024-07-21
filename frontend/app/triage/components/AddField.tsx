import React, { useState } from "react";

interface DisplayCols {
  [key: string]: boolean;
}

interface AddFieldProps {
  displayCols: DisplayCols;
  handleAddField: (fieldName: string) => void;
  handleSelectAll: (selectAll: boolean) => void;
}

const AddField: React.FC<AddFieldProps> = ({ displayCols, handleAddField, handleSelectAll }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  const allSelected = Object.values(displayCols).every((b) => b);

  return (
    <div className="relative inline-block">
      <button
        className={`px-2 py-1 m-1 rounded-lg shadow-md ${
          isOpen
            ? "bg-blue-600 hover:bg-blue-700"
            : "bg-blue-700 hover:bg-blue-800"
        } text-white font-semibold focus:outline-none focus:ring-blue-500`}
        onClick={toggleDropdown}
      >
        <svg
          className={`w-5 h-5 ${isOpen ? "transform rotate-180" : ""}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div
          className="absolute top-12 left-0 rounded-lg w-64 h-[50vh] overflow-auto scroll-smooth shadow-lg p-4 border border-solid border-gray-300 bg-white"
          style={{ overflowY: "auto", zIndex: 10 }}
          onMouseEnter={() => setIsOpen(true)}
          onMouseLeave={() => setIsOpen(false)}
        >
          <div className="flex items-center m-1 text-md">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={() => handleSelectAll(!allSelected)}
              className="form-checkbox h-4 w-4 text-blue-600"
            />
            <label className="ml-2 text-gray-700 font-medium">Select All</label>
          </div>
          <hr className="my-2" />
          {Object.entries(displayCols).map(([fieldName, b]) => (
            <div key={fieldName} className="flex items-center m-1 text-md">
              <input
                type="checkbox"
                checked={b}
                onChange={() => handleAddField(fieldName)}
                className="form-checkbox h-4 w-4 text-blue-600"
              />
              <label
                className={`ml-2 ${b ? "text-blue-700 font-semibold" : "text-gray-700"}`}
              >
                {fieldName}
              </label>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AddField;
