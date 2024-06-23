import Link from "next/link";
import React from "react";
import { HiOutlineArrowLeft } from "react-icons/hi";

interface TriageHeaderProps {
  doc_id: string | null;
}

const TriageHeader: React.FC<TriageHeaderProps> = ({ doc_id }) => {
  return (
    <div className="flex justify-between bg-gradient-to-r from-blue-300 to-gray-200 rounded-md sm:p-2 md:p-4 lg:p-6 xl:p-8 shadow-lg">
      <Link href="/" className="hover:underline flex items-center">
        <HiOutlineArrowLeft className="mr-2 h-6 w-6" />
        Back
      </Link>

      <h1 className="text-2xl font-bold text-black-900 tracking-wide">
        Invoice Parsing
      </h1>

      <div className="flex items-center">
        <p className="text-gray-800 mr-2">Document ID:</p>
        <p className="text-gray-600 font-semibold">{doc_id}</p>
      </div>
    </div>
  );
}

export default TriageHeader;
