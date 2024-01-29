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
      formData.append("file", file);

      try {
        setIsLoading(true); // Set loading state to true

        const response = await fetch(
          "http://127.0.0.1:8000/db_connect/upload_doc/",
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
    <div className="flex justify-between items-center bg-gradient-to-r from-blue-300 to-gray-200 rounded-md sm:p-2 md:p-4 lg:p-6 xl:p-8 shadow-lg">
      <div className="flex items-center">
        <Image
          src={"/Tally-Logo.webp"}
          alt="Image"
          width={100}
          height={100}
          className="rounded-full"
        />
      </div>
      <h1 className="text-2xl font-bold text-black">
        Tally-AI Invoice Parsing
      </h1>
      <div className="flex items-center text-xs">
        <label
          htmlFor="file-upload"
          className={`cursor-pointer bg-gradient-to-r from-cyan-300 to-blue-500 hover:bg-gradient-to-bl text-black sm:p-0 md:p-1 lg:p-2 xl:p-3 rounded-xl flex items-center transition duration-100 ${
            isLoading ? "opacity-50 pointer-events-none" : ""
          }`}
        >
          {isLoading ? (
            <p className="m-1">Uploading...</p>
          ) : (
            <p className="m-1">+ Upload File</p>
          )}
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
