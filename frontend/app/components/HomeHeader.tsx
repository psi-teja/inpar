"use client";
import Link from "next/link";
import Image from "next/image";
import { useState, ChangeEvent } from "react";
import BACKEND_URLS from "../BackendUrls";

function HomeHeader() {
  const [isLoading, setIsLoading] = useState(false);

  const handleFileUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      const formData = new FormData();
      formData.append("document", file);

      try {
        setIsLoading(true); // Set loading state to true

        const response = await fetch(
          BACKEND_URLS.uploadDocUrl,
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
      <Link href="https://github.com/psi-teja/inpar">
          <Image
            src="/GitHub.png"
            alt="GitHub Logo"
            width={50}
            height={50}
            className="rounded-full"
            priority
          />
      </Link>
    </div>
    <h1 className="text-2xl font-bold text-teal-900 sm:p-0 md:p-1 lg:p-2 xl:p-3">
      Invoice Parsing
    </h1>
    <div className="flex items-center">
      <label
        htmlFor="file-upload"
        className={`cursor-pointer bg-gradient-to-r from-cyan-300 to-blue-500 hover:bg-gradient-to-bl text-black sm:p-0 md:p-1 lg:p-2 xl:p-3 rounded-xl flex items-center transition duration-100 ${isLoading ? "opacity-50 pointer-events-none" : ""}`}
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
          />
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
