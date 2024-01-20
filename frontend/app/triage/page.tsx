"use client"
import { useSearchParams } from 'next/navigation';
import DocViewer from '@/app/triage/components/DocViewer';
import ExtractedFields from '@/app/triage/components/ExtractedFields';
import TriageHeader from '@/app/triage/components/TriageHeader';
export default async function Triage() {
  const searchParams = useSearchParams();
  const doc_id = searchParams.get('doc_id');
  const file_path = searchParams.get('file_path');

  return (
    <main className='h-[100vh]'>
      <TriageHeader doc_id={doc_id} />
      <div className="flex">
        <DocViewer doc_id={doc_id} />
        <ExtractedFields doc_id={doc_id} />
      </div>
    </main>

  );
}
