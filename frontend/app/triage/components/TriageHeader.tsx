import Link from "next/link";
import React from "react";
import { HiOutlineArrowLeft } from "react-icons/hi";

interface TriageHeaderProps {
  doc_id: string | null;
}

const TriageHeader: React.FC<TriageHeaderProps> = ({ doc_id }) => {
  return (
    <div className="flex justify-between items-center shadow bg-blue-600 sm:p-1 md:p-2 lg:p-3 xl:p-3">
      <Link href="/" className="text-white hover:underline flex items-center">
        <HiOutlineArrowLeft className="mr-2 h-6 w-6" />
        Back
      </Link>

      <h1 className="text-2xl font-bold text-black-900 tracking-wide">
        Invoice Parsing
      </h1>

      <div className="flex items-center">
        <p className="text-gray-800 mr-2">Document ID:</p>
        <p className="text-gray-800 font-semibold">{doc_id}</p>
      </div>
    </div>
  );
}

export default TriageHeader;
