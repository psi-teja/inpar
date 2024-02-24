import React, { ChangeEvent } from "react";

interface PdfToolsProps {
  scale: number;
  handlePageNumberChange: (event: ChangeEvent<HTMLInputElement>) => void;
  pageNumber: number;
  numPages: number | undefined;
  downloadFile: (file: any, docType: string) => void;
  file: string;
  handleScaleChange: (event: ChangeEvent<HTMLInputElement>) => void;
  docType: string;
}

const PdfTools: React.FC<PdfToolsProps> = ({
  scale,
  handlePageNumberChange,
  pageNumber,
  numPages,
  downloadFile,
  file,
  handleScaleChange,
  docType,
}) => {
  return (
    <div className="p-1 flex justify-between text-xs bg-slate-300">
      <input
        className="sm:w-20 md:w-30 lg:w-40 xl:w-50"
        id="scaleSlider"
        type="range"
        min="0.1"
        max="2"
        step="0.1"
        value={scale}
        onChange={handleScaleChange}
      />
      <div className="flex items-center">
        <input
          type="number"
          value={pageNumber}
          onChange={handlePageNumberChange}
          className="w-16 text-center border-1 rounded focus:outline-none focus:border-blue-500"
        />
        / {numPages}
      </div>
      <div
        className="hover:bg-gray-200 rounded mr-4"
        onClick={() => downloadFile(file, docType)}
      >
        <svg
          className="h-5 w-5 text-black "
          width="24"
          height="24"
          viewBox="0 0 24 24"
          stroke-width="2"
          stroke="currentColor"
          fill="none"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          {" "}
          <path stroke="none" d="M0 0h24v24H0z" />{" "}
          <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2" />{" "}
          <polyline points="7 11 12 16 17 11" />{" "}
          <line x1="12" y1="4" x2="12" y2="16" />
        </svg>
      </div>
    </div>
  );
};

export default PdfTools;
