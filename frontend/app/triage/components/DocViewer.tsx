import React, { useEffect, useState } from "react";
import FileSaver from 'file-saver';
import PdfViewer from "./PdfViewer";


interface DocViewerProps {
  doc_id: string | null;
  boxLocation: Record<string, any> | null; // Define the type for boxLocation
  viewType: string; // Define the type for viewType
}

const DocViewer: React.FC<DocViewerProps> = ({ doc_id, boxLocation, viewType }) => {
  const fileUrl = `http://localhost:8000/db_connect/get-document/${doc_id}`;

  const downloadFile = (file: string, docType: string) => {
    fetch(file)
      .then((response) => response.blob())
      .then((blob) => {
        FileSaver.saveAs(blob, `${doc_id}.${docType}`);
      })
      .catch((error) => {
        console.error('Error downloading file:', error);
      });
  };

  return (
    <div
      className={`${
        viewType === "DocInfo" ? "w-[70vw]" : ""
      } border border-blue-400`}
    >
        <PdfViewer
          file={fileUrl}
          boxLocation={boxLocation}
          viewType={viewType}
          downloadFile={downloadFile}
        />
    </div>
  );
}

export default DocViewer;
