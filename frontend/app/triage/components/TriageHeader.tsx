"use client";
import Link from "next/link";
import React, { useState } from "react";
import { HiOutlineArrowLeft } from "react-icons/hi";
import { FaSpinner } from "react-icons/fa";

interface TriageHeaderProps {
  doc_id: string | null;
}

const TriageHeader: React.FC<TriageHeaderProps> = ({ doc_id }) => {
  const [homeReady, setHomeReady] = useState<boolean>(false);

  if (homeReady) {
    return (
      <div className="flex items-center justify-center h-screen">
        <FaSpinner className="animate-spin text-4xl text-gray-600" />
      </div>
    );
  }

  return (
    <div className="flex justify-between items-center bg-gradient-to-r from-blue-600 to-cyan-300 p-4 shadow-lg">
      <Link href="/" className="text-white hover:underline flex items-center"
        onClick={() => setHomeReady(true)}>
        <HiOutlineArrowLeft className="mr-2 h-6 w-6" />
        Back
      </Link>

      <h1 className="text-3xl font-extrabold text-white tracking-tight leading-tight">
        InPar
      </h1>

      <div className="flex items-center text-cyan-900">
        <p className="mr-2">Document ID:</p>
        <p className="font-semibold">{doc_id}</p>
      </div>
    </div>
  );
}

export default TriageHeader;
