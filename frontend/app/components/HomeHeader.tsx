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
        setIsLoading(true);

        const response = await fetch(BACKEND_URLS.uploadDocUrl, {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          console.log("File uploaded successfully");
          window.location.reload();
        } else {
          console.error("Failed to upload file");
        }
      } catch (error) {
        console.error("Error during file upload:", error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="flex justify-between items-center bg-gradient-to-br from-blue-500 to-cyan-400 p-4 shadow-lg">
      <div className="flex items-center">
        <Link href="https://github.com/psi-teja/inpar">
          <div className="rounded-full overflow-hidden cursor-pointer">
            <Image
              src="/GitHub.png"
              alt="GitHub Logo"
              width={50}
              height={50}
              priority
            />
          </div>
        </Link>
      </div>
      <h1 className="text-3xl font-extrabold text-white tracking-tight leading-tight">
        InPar
      </h1>
      <div className="flex items-center ">
        <label
          htmlFor="file-upload"
          className={`cursor-pointer active:border-2 bg-gradient-to-r from-blue-800 to-cyan-500 hover:bg-gradient-to-bl text-white px-4 py-2 rounded-xl flex items-center transition duration-300 ${isLoading ? "opacity-50 pointer-events-none" : ""}`}
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
          accept=".pdf, .doc, .docx, .jpeg, .png, .jpg, .webp"
          className="hidden" 
          onChange={handleFileUpload}
        />
      </div>
    </div>
  );
}

export default HomeHeader;
