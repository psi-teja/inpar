"use client";

import { useState, useEffect, useRef } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/esm/Page/TextLayer.css";
import "react-pdf/dist/esm/Page/AnnotationLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

function PdfViewer({ file , boxLocation }: { file: string, boxLocation: object }) {
  const [numPages, setNumPages] = useState<number>();
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [scale, setScale] = useState(1);
  const [pdfDim, setPdfDim] = useState({ width: 0, height: 0 });
  const boundingBoxRef =  useRef<HTMLDivElement | null>(null);
  const [prevPageNumber, setPrevPageNumber] = useState<number | null>(null);
  

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

    const { ltwh, pageNo } = boxLocation || {};

    (pageNo)?setPageNumber(pageNo): 1;

    if (boundingBoxRef.current) {
      boundingBoxRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "center",
      });
    }
  }, [boxLocation]);

  let { ltwh, pageNo } = boxLocation || {};
  const [left, top, width, height] = ltwh || [];

  const renderAnnotations = () => {
    if (!boxLocation || !pdfDim.width || !pdfDim.height || pageNo != pageNumber) return null;
    if (!ltwh) return null

    const [left, top, width, height] = ltwh;

    const scaledLeft = left * scale * pdfDim.width;
    const scaledTop = top * scale * pdfDim.height;
    const scaledWidth = width * scale * pdfDim.width;
    const scaledHeight = height * scale * pdfDim.height;

    return (
      <div
        ref={boundingBoxRef}
        className="bg-red-400 bg-opacity-40"
        style={{
          position: "absolute",
          left: `${scaledLeft-1}px`,
          top: `${scaledTop-1}px`,
          width: `${scaledWidth+4}px`,
          height: `${scaledHeight+2}px`,
          border: "1px solid red", // You can customize the border style
          pointerEvents: "none", // Ensure the annotation does not interfere with user interactions
        }}
      />
    );
  };

  function onDocumentLoadSuccess({ numPages }: { numPages: number }): void {
    setNumPages(numPages);
  }

  const handlePreviousPage = () => setPageNumber(Math.max(1, pageNumber - 1));
  const handleNextPage = () => {
    if (numPages) {
      setPageNumber(Math.min(numPages, pageNumber + 1));
    }
    ltwh = [0,0,0,0]
  };

  const handleScaleChange = (e) => {
    const newScale = parseFloat(e.target.value);
    setScale(newScale);
  };

  const handlePageNumberChange = (e) => {
    const newPageNumber = Math.min(
      numPages ? numPages : 1,
      Math.max(1, parseInt(e.target.value, 10))
    );
    setPageNumber(newPageNumber);
  };

  return (
    <div>
      <div className="flex justify-between items-center text-xs bg-slate-300">
        <div className="flex items-center">
          <button
            className="p-1 m-1 btn-nav hover:bg-blue-600 rounded hover:text-white"
            onClick={handlePreviousPage}
            disabled={pageNumber === 1}
          >
            &lt;&lt; Previous
          </button>
          <div className="m-1 ml-8 rounded-lg border-2 hover:border-blue-400">
            <button
              onClick={() =>
                handleScaleChange({ target: { value: scale - 0.1 } })
              }
              className="text-lg hover:bg-blue-600 hover:text-white bg-gray-200 px-2 mr-2 rounded-l-md cursor-pointer transition-all duration-300 ease-in-out"
            >
              -
            </button>
            <input
              id="scaleSlider"
              type="range"
              min="0.1"
              max="3"
              step="0.1"
              value={scale}
              onChange={handleScaleChange}
            />
            <button
              onClick={() => 
                handleScaleChange({ target: { value: scale + 0.1 } })
              }
              className="text-lg hover:bg-blue-600 hover:text-white bg-gray-200 px-2 ml-2 rounded-r-md cursor-pointer transition-all duration-300 ease-in-out"
            >
              +
            </button>
          </div>
        </div>
        <span className="flex items-center">
          <input
            type="number"
            value={pageNumber}
            onChange={handlePageNumberChange}
            className="w-16 p-1 text-center border-1 rounded focus:outline-none focus:border-blue-500"
          />
          / {numPages}
        </span>
        <button
          className="p-1 m-1 btn-nav hover:bg-blue-700 rounded hover:text-white"
          onClick={handleNextPage}
          disabled={pageNumber === numPages}
        >
          Next &gt;&gt;
        </button>
      </div>
      <div className="h-[85vh] overflow-y-auto shadow-xl">
        <Document file={file} onLoadSuccess={onDocumentLoadSuccess}>
          <Page pageNumber={pageNumber} scale={scale} >
          {renderAnnotations()}
          </Page>
        </Document>
      </div>
    </div>
  );
}

export default PdfViewer;
