import React from 'react';

interface ToggleViewProps {
    viewType: string | null;
    handleChangeView: (viewType: string) => void;
}

const ToggleView: React.FC<ToggleViewProps> = ({
    viewType,
    handleChangeView
}) => {
    const buttonBaseClass = "text-gray-800 px-3 py-1 rounded-t border-r border-black focus:outline-none";

    const activeButtonClass = "bg-blue-500 text-white";
    const inactiveButtonClass = "bg-gray-300 text-gray-700 hover:bg-blue-300 hover:text-gray-800";

    return (
        <div className="flex ml-3">
            <button
                className={`${buttonBaseClass} ${viewType === "General" ? activeButtonClass : inactiveButtonClass}`}
                onClick={() => handleChangeView("General")}
            >
                General
            </button>
            <button
                className={`${buttonBaseClass} ${viewType === "Items" ? activeButtonClass : inactiveButtonClass}`}
                onClick={() => handleChangeView("Items")}
            >
                Items
            </button>
            <button
                className={`${buttonBaseClass} ${viewType === "Ledgers" ? activeButtonClass : inactiveButtonClass}`}
                onClick={() => handleChangeView("Ledgers")}
            >
                Ledgers
            </button>
        </div>
    )
}

export default ToggleView;
