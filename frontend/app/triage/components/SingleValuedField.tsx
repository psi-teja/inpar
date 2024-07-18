import React from "react";

interface SingleValuedFieldProps {
  fieldName: string;
  fieldValue: Record<string, any>;
  selectedField: string | null;
  handleFieldClick: (
    fieldName: string,
    index: number | null,
    colName: string | null,
    location: Record<string, any>
  ) => void;
  handleSingleValuedFieldChange: (
    fieldName: string,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => void;
}

const SingleValuedField: React.FC<SingleValuedFieldProps> = ({
  fieldName,
  fieldValue,
  selectedField,
  handleFieldClick,
  handleSingleValuedFieldChange,
}) => {

  if (fieldValue.location?.pageNo == 0 && fieldValue.text == "") return;

  return (
    <div
      className={`m-2 bg-gray-200 border border-gray-400 sm:text-xs md:text-xs lg:text-lg xl:text-lg cursor-default p-2 rounded-md shadow-md transition duration-300 ease-in-out hover:shadow-lg ${selectedField === fieldName ? "bg-red-200" : "bg-white"
        }`}
      key={fieldName}
      onFocus={() => {
        handleFieldClick(fieldName, null, null, fieldValue.location);
      }}
      onClick={() => {
        handleFieldClick(fieldName, null, null, fieldValue.location);
      }}
    >
      <div className="flex items-center justify-between">
        <p className="font-semibold m-1 text-indigo-700 text-center">
          {fieldName}
        </p>

        {fieldValue.location?.pageNo !== 0 && (
          <button
            onClick={(e) =>
              handleSingleValuedFieldChange(
                fieldName,
                fieldValue.text,
                null,
                "del bbox"
              )
            }
            disabled={fieldName !== selectedField}
            className="relative"
          >
            <img
              src="rect.png" // Replace with the actual path to your PNG image
              alt="Draw Box"
              className="h-4 w-5 m-2" // Adjust the height and width of the image as needed
            />
            {fieldName === selectedField && (
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

      <textarea
        className={`text-gray-800 bg-blue-50 rounded-md border border-blue-300 p-2 focus:outline-none w-full ${selectedField === fieldName ? "border border-red-300" : ""
          }`}
        value={fieldValue.text}
        placeholder="AI couldn't extract"
        onChange={(e) => {
          handleSingleValuedFieldChange(
            fieldName,
            e.target.value,
            fieldValue.location,
            "update value"
          );
        }}
        rows={1}
        wrap="soft"
      />

    </div>
  );
};

export default SingleValuedField;
