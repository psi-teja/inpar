'use client'
import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import InteractiveSpace from './components/InteractiveSpace';
import TriageHeader from '@/app/triage/components/TriageHeader';

export default function Triage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <TriageContent />
    </Suspense>
  );
}

function TriageContent() {
  const searchParams = useSearchParams();
  const doc_id = searchParams.get('doc_id');
  const json_type = searchParams.get('json_type')

  return (
    <main className=''>
      <TriageHeader doc_id={doc_id} />
      <InteractiveSpace doc_id={doc_id} json_type={json_type}/>
    </main>
  );
}
