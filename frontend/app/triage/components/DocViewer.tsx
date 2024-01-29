import React, { useEffect, useState } from "react";
import PdfViewer from "./PdfViewer";
import Image from "next/image";

function DocViewer({ doc_id, boxLocation }) {
  const [isPdf, setIsPdf] = useState(false);
  const fileUrl = `http://localhost:8000/db_connect/get-document/${doc_id}`;

  useEffect(() => {
    const checkIsPdf = async () => {
      try {
        const response = await fetch(fileUrl);
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.toLowerCase().includes("application/pdf")) {
          setIsPdf(true);
        }
      } catch (error) {
        console.error("Error checking file type:", error);
      }
    };
    checkIsPdf();
  }, [fileUrl]);

  return (
    <div className="w-[70vw] border border-r-blue-600">
      {isPdf ? (
        <PdfViewer file={fileUrl} boxLocation={boxLocation}/>
      ) : (
        <Image
          className="w-full max-h-[89vh] object-contain"
          src={fileUrl}
          alt="Image"
          width={1000}
          height={1000}
        />
      )}
    </div>
  );
}

export default DocViewer;
