"use client";

import { useState, useEffect, useRef } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/TextLayer.css";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";
import PdfTools from "./PdfTools";

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

interface PdfViewerProps {
  file: string;
  boxLocation: Record<string, any> | null;
  viewType: string;
  downloadFile: (file: any, docType: string) => void;
}

const PdfViewer: React.FC<PdfViewerProps> = ({
  file,
  boxLocation,
  viewType,
  downloadFile,
}) => {
  const [numPages, setNumPages] = useState<number>();
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [scale, setScale] = useState(1);
  const [pdfDim, setPdfDim] = useState({ width: 0, height: 0 });
  const boundingBoxRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const fetchPdfDimensions = async () => {
      const pdfDoc = await pdfjs.getDocument(file).promise;
      const firstPage = await pdfDoc.getPage(1);
      const { width, height } = firstPage.getViewport({ scale: 1 });
      setPdfDim({ width, height });
    };
    fetchPdfDimensions();
  }, [file]);

  useEffect(() => {
    if (boundingBoxRef.current) {
      boundingBoxRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "center",
      });
    }
  }, [boxLocation]);

  const renderAnnotations = () => {
    if (boxLocation === null) {
      return null;
    }

    const pageNo = boxLocation.pageNo;

    if (
      !boxLocation ||
      !pdfDim.width ||
      !pdfDim.height ||
      pageNo !== pageNumber
    ) {
      return null;
    }

    const [left, top, width, height] = boxLocation.ltwh || [0, 0, 0, 0];

    const scaledLeft = left * scale * pdfDim.width;
    const scaledTop = top * scale * pdfDim.height;
    const scaledWidth = width * scale * pdfDim.width;
    const scaledHeight = height * scale * pdfDim.height;

    const boundingBoxStyle =
      "absolute bg-blue-300 bg-opacity-20 pointer-events-none";

    return (
      <div
        ref={boundingBoxRef}
        className={boundingBoxStyle}
        style={{
          left: `${scaledLeft - 1}px`,
          top: `${scaledTop - 1}px`,
          width: `${scaledWidth + 4}px`,
          height: `${scaledHeight + 4}px`,
          boxShadow: "0 0 10px rgba(255, 0, 0, 0.6)",
          borderRadius: "3px",
          border: "1px solid  #f00",
        }}
      />
    );
  };

  function onDocumentLoadSuccess({ numPages }: { numPages: number }): void {
    setNumPages(numPages);
  }

  const handleScaleChange = (e: { target: { value: string; }; }) => {
    const newScale = parseFloat(e.target.value);
    setScale(newScale);
  };

  const handlePageNumberChange = (e: { target: { value: string; }; }) => {
    const newPageNumber = Math.min(
      numPages ? numPages : 1,
      Math.max(1, parseInt(e.target.value, 10))
    );
    setPageNumber(newPageNumber);
  };

  return (
    <div className="shadow-lg">
      <PdfTools
        docType={"pdf"}
        scale={scale}
        handlePageNumberChange={handlePageNumberChange}
        pageNumber={pageNumber}
        numPages={numPages}
        downloadFile={downloadFile}
        file={file}
        handleScaleChange={handleScaleChange}
      />
      <div
        className={`${
          viewType === "DocInfo" ? "w-[70vw] h-[87.5vh]" : "h-[50vh]"
        } overflow-y-auto`}
      >
        <Document file={file} onLoadSuccess={onDocumentLoadSuccess}>
          <Page pageNumber={pageNumber} scale={scale}>
            {renderAnnotations()}
          </Page>
        </Document>
      </div>
    </div>
  );
};

export default PdfViewer;
