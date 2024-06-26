import React, { useState, useEffect, useRef } from "react";
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
  handleSingleValuedFieldChange: (
    fieldName: string | null,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => void; // Add a new prop to update the boxLocation
  handleNestedFieldChange: (
    fieldType: string,
    index: number | null,
    fieldName: string | null,
    value: string | null,
    location: Record<string, any> | null,
    instruction: string
  ) => void;
  selectedField: string | null;
  selectedRow: number | null;
  dataChanged: boolean;
}

const PdfViewer: React.FC<PdfViewerProps> = ({
  file,
  boxLocation,
  viewType,
  downloadFile,
  handleSingleValuedFieldChange,
  handleNestedFieldChange,
  selectedField,
  selectedRow,
}) => {
  const [numPages, setNumPages] = useState<number>();
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [scale, setScale] = useState(1);
  const [pdfDim, setPdfDim] = useState({ width: 0, height: 0 });
  const [drawingBox, setDrawingBox] = useState<boolean>(false);
  const [startX, setStartX] = useState<number>(0);
  const [startY, setStartY] = useState<number>(0);
  const [endX, setEndX] = useState<number>(0);
  const [endY, setEndY] = useState<number>(0);
  const boundingBoxRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<HTMLDivElement | null>(null);
  // const [showScrollbar, setShowScrollbar] = useState<boolean>(false);

  const viewerLoc = viewerRef.current?.getBoundingClientRect();

  useEffect(() => {
    const fetchPdfDimensions = async () => {
      const pdfDoc = await pdfjs.getDocument(file).promise;
      const firstPage = await pdfDoc.getPage(1);
      const { width, height } = firstPage.getViewport({ scale: 1 });
      setPdfDim({ width, height });
    };
    fetchPdfDimensions();
  }, [file, viewType]);

  useEffect(() => {
    viewerLoc ? setScale(viewerLoc.width / pdfDim.width) : "";
  }, [pdfDim]);

  useEffect(() => {
    if (boundingBoxRef.current) {
      boundingBoxRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "center",
      });
    }
    if (boxLocation){
      setPageNumber(boxLocation.pageNo)
    }
  }, [boxLocation]);

  const renderAnnotations = () => {
    if (boxLocation === null) {
      if (startX == 0 && startY == 0) {
        return null;
      } else {
        return selectedField && newRectangle();
      }
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
      <div>
        <div
          ref={boundingBoxRef}
          className={boundingBoxStyle}
          style={{
            left: `${scaledLeft}px`,
            top: `${scaledTop}px`,
            width: `${scaledWidth}px`,
            height: `${scaledHeight}px`,
            boxShadow: "0 0 10px rgba(255, 0, 0, 0.6)",
            borderRadius: "3px",
            border: "1px solid  #f00",
          }}
        />
      </div>
    );
  };

  const newRectangle = () => {
    const boundingBoxStyle =
      "absolute bg-blue-300 bg-opacity-20 pointer-events-none";

    // Get the reference to the page div
    const pageDiv = viewerRef.current;

    if (!pageDiv) return null;

    // Calculate the scroll offset of the page div
    const scrollOffsetX = pageDiv.scrollLeft;
    const scrollOffsetY = pageDiv.scrollTop;

    return (
      <div
        className={boundingBoxStyle}
        style={{
          left: `${Math.min(startX, endX) + scrollOffsetX}px`, // Adjust for horizontal scroll
          top: `${Math.min(startY, endY) + scrollOffsetY}px`, // Adjust for vertical scroll
          width: `${Math.abs(startX - endX)}px`,
          height: `${Math.abs(startY - endY)}px`,
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

  const handleScaleChange = (e: { target: { value: string } }) => {
    const newScale = parseFloat(e.target.value);
    setScale(newScale);
  };

  const handlePageNumberChange = (e: { target: { value: string } }) => {
    const newPageNumber = Math.min(
      numPages ? numPages : 1,
      Math.max(1, parseInt(e.target.value, 10))
    );
    setPageNumber(newPageNumber);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    setDrawingBox(true);
    if (viewerLoc && !boxLocation) {
      setStartX(e.clientX - viewerLoc.left);
      setStartY(e.clientY - viewerLoc.top);
      setEndX(e.clientX - viewerLoc.left);
      setEndY(e.clientY - viewerLoc.top);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!boxLocation && viewerLoc && drawingBox) {
      setEndX(e.clientX - viewerLoc.left);
      setEndY(e.clientY - viewerLoc.top);
    }
  };

  const handleMouseUp = async () => {
    if (!boxLocation) {
        setDrawingBox(false);

        if (viewerLoc) {
            // Calculate the scroll offset of the page div
            const scrollOffsetX = viewerRef.current?.scrollLeft || 0;
            const scrollOffsetY = viewerRef.current?.scrollTop || 0;

            // Calculate the position of the rectangle relative to the scrolled viewport
            const left = Math.min(startX, endX) + scrollOffsetX;
            const top = Math.min(startY, endY) + scrollOffsetY;
            const width = Math.abs(endX - startX);
            const height = Math.abs(endY - startY);
            const scaledLeft = left / pdfDim.width / scale;
            const scaledTop = top / pdfDim.height / scale;
            const scaledWidth = width / pdfDim.width / scale;
            const scaledHeight = height / pdfDim.height / scale;

            if (viewType === "DocInfo") {
                handleSingleValuedFieldChange(
                    selectedField,
                    null,
                    {
                        pageNo: pageNumber,
                        ltwh: [scaledLeft, scaledTop, scaledWidth, scaledHeight],
                    },
                    "add bbox"
                );
            } else {
                handleNestedFieldChange(
                    "Table",
                    selectedRow,
                    selectedField,
                    null,
                    {
                        pageNo: pageNumber,
                        ltwh: [scaledLeft, scaledTop, scaledWidth, scaledHeight],
                    },
                    "add bbox"
                );
            }

            // Crop the selected area as an image
            const canvas = document.createElement("canvas");
            const context = canvas.getContext("2d");
            canvas.width = width;
            canvas.height = height;
            context?.drawImage(
                document.getElementsByTagName("canvas")[0],
                left,
                top,
                width,
                height,
                0,
                0,
                width,
                height
            );
            const imageDataUrl = canvas.toDataURL();
            // console.log(left, top)
            // console.log(document.getElementsByTagName("canvas")[0])
            // console.log(imageDataUrl)

            // Convert the data URL to a blob
            const blob = await fetch(imageDataUrl).then((res) => res.blob());

            // Send the image to the OCR endpoint
            const formData = new FormData();
            formData.append("file", blob, "cropped_image.png");

            try {
                const response = await fetch("http://localhost:8000/ocr/", {
                    method: "POST",
                    body: formData,
                });
                const data = await response.json();
                console.log("OCR Result:", data.text);
            } catch (error) {
                console.error("Error during OCR:", error);
            }
        }

        setStartX(0);
        setStartY(0);
    }
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
        } overflow-auto scroll-smooth ${(!boxLocation && selectedField)? "cursor-crosshair":""}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        ref={viewerRef}
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
