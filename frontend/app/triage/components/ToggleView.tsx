import React from 'react';

interface ToggleViewProps {
    viewType: string | null;
    handleChangeView: (viewType: string) => void;
}

const ToggleView: React.FC<ToggleViewProps> = ({
    viewType,
    handleChangeView
}) => {
    const buttonBaseClass =
        'px-2 py-1 rounded-t-lg focus:outline-none transition duration-200 ease-in-out';

    const activeButtonClass =
        'bg-blue-500 text-white shadow-md transform scale-105';
    const inactiveButtonClass =
        'bg-gray-300 text-gray-700 hover:bg-blue-300 hover:text-white';

    return (
      <div className={`flex ${viewType == "General" || viewType== "ROI"? "ml-3" : "ml-3"}`}>
        <button
          className={`${buttonBaseClass} ${
            viewType === "General" ? activeButtonClass : inactiveButtonClass
          }`}
          onClick={() => handleChangeView("General")}
        >
          General
        </button>
        <button
          className={`${buttonBaseClass} ${
            viewType === "Items" ? activeButtonClass : inactiveButtonClass
          }`}
          onClick={() => handleChangeView("Items")}
        >
          Items
        </button>
        <button
          className={`${buttonBaseClass} ${
            viewType === "Ledgers" ? activeButtonClass : inactiveButtonClass
          }`}
          onClick={() => handleChangeView("Ledgers")}
        >
          Ledgers
        </button>
        <button
          className={`${buttonBaseClass} ${
            viewType === "ROI" ? activeButtonClass : inactiveButtonClass
          }`}
          onClick={() => handleChangeView("ROI")}
        >
          ROI
        </button>
      </div>
    );
}

export default ToggleView;
