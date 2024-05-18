"use client";
import Link from "next/link";
import Image from "next/image";
import { useState } from "react";

function HomeHeader() {
  const [isLoading, setIsLoading] = useState(false);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      const formData = new FormData();
      formData.append("document", file);

      try {
        setIsLoading(true); // Set loading state to true

        const response = await fetch(
          "https://loqc5abj3t6z2crtwezyyxcsae0zsxmz.lambda-url.ap-south-1.on.aws/upload/doc",
          {
            method: "POST",
            body: formData,
          }
        );

        if (response.ok) {
          console.log("File uploaded successfully");
          window.location.reload();
        } else {
          console.error("Failed to upload file");
        }
      } catch (error) {
        console.error("Error during file upload:", error);
      } finally {
        setIsLoading(false); // Set loading state back to false
      }
    }
  };

  return (
    <div className="flex justify-between bg-gradient-to-r from-blue-300 to-gray-200 rounded-md sm:p-2 md:p-4 lg:p-6 xl:p-8 shadow-lg">
      <div className="flex items-center">
        <Image
          src={"/Tally-Logo.webp"}
          alt="Image"
          width={100}
          height={100}
          className="rounded-full"
        />
      </div>
      <h1 className="text-2xl font-bold text-teal-900 sm:p-0 md:p-1 lg:p-2 xl:p-3">
          Tally-AI Invoice Parsing
      </h1>
      <div className="flex items-center">
        <label
          htmlFor="file-upload"
          className={`cursor-pointer bg-gradient-to-r from-cyan-300 to-blue-500 hover:bg-gradient-to-bl text-black sm:p-0 md:p-1 lg:p-2 xl:p-3 rounded-xl flex items-center transition duration-100 ${
            isLoading ? "opacity-50 pointer-events-none" : ""
          }`}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 mr-2"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 6v6m0 0v6m0-6h6m-6 0H6"
            ></path>
          </svg>
          {isLoading ? "Uploading..." : "Upload File"}
        </label>
        <input
          id="file-upload"
          type="file"
          accept=".pdf, .doc, .docx, .jpeg, .png, .jpg"
          className="hidden"
          onChange={handleFileUpload}
        />
      </div>
    </div>
  );
}

export default HomeHeader;
