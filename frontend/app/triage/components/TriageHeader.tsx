import Link from "next/link";
import React from "react";
import { HiOutlineArrowLeft } from "react-icons/hi";

function TriageHeader({ doc_id }) {
  return (
    <div className="flex justify-between items-center bg-gradient-to-r from-blue-300 to-blue-700 sm:p-2 md:p-4 lg:p-6 xl:p-8">
  <Link href="/" className="text-black hover:underline flex items-center">
    <HiOutlineArrowLeft className="mr-2 h-6 w-6" />
    Back
  </Link>

  <h1 className="text-2xl font-bold text-black-900 tracking-wide">Triage</h1>

  <div className="flex items-center">
    <p className="text-gray-800 mr-2">Document ID:</p>
    <p className="text-blue-100 font-semibold">{doc_id}</p>
  </div>
</div>

  );
}

export default TriageHeader;
