'use client'
import { useSearchParams } from 'next/navigation';
import InteractiveSpace from './components/InteractiveSpace';
import TriageHeader from '@/app/triage/components/TriageHeader';

export default async function Triage() {
  const searchParams = useSearchParams();
  const doc_id = searchParams.get('doc_id');
  

  return (
    <main className='h-[100vh]'>
      <TriageHeader doc_id={doc_id} />
      <InteractiveSpace doc_id={doc_id} />
    </main>
  );
}
