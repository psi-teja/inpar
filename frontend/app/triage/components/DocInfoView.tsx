import React from "react";
interface DocInfoProps {
  fieldName: string;
  fieldValue: Record<string, any>;
  selectedField: string | null;
  handleFieldClick: (fieldName: string, boxLocation: Record<string, any>) => void;
  handleDocInfoFieldChange: (fieldName: string, fieldValue: string) => void;
}

const DocInfo: React.FC<DocInfoProps> = ({
  fieldName,
  fieldValue,
  selectedField,
  handleFieldClick,
  handleDocInfoFieldChange,
}) => {
  return (
    <div
      className={`m-2 bg-gray-200 border border-gray-400 sm:text-xs md:text-xs lg:text-lg xl:text-lg cursor-default p-2 rounded-md shadow-md transition duration-300 ease-in-out hover:shadow-lg ${
        selectedField === fieldName ? "bg-red-200" : "bg-white"
      }`}
      key={fieldName}
      onClick={() => {
        handleFieldClick(fieldName, fieldValue.location);
      }}
    >
      <p className="font-semibold mb-2 text-indigo-700">{fieldName}</p>
      <textarea
        className={`text-gray-800 bg-blue-50 rounded-md border border-blue-300 p-2 focus:outline-none w-full ${selectedField === fieldName? "border border-red-300": ""}` }
        value={fieldValue.text}
        style={{
          minHeight: "40px",
          height: "auto",
          maxHeight: "200px",
        }}
        onChange={(e) => {
          handleDocInfoFieldChange(fieldName, e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = e.target.scrollHeight + "px";
        }}
      />
    </div>
  );
}

export default DocInfo;
