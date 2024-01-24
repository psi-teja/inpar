import {useState} from 'react';
import DocViewer from '@/app/triage/components/DocViewer';
import ExtractedFields from '@/app/triage/components/ExtractedFields';

function InteractiveSpace({doc_id}) {
const [boxLocation, setBoxLocation] = useState({'ltwh':[0,0,0,0], 'pageNo':0});

  const changeBox = (location) => {
    setBoxLocation(location)
  }
  
    return (
    <div className="flex">
        <DocViewer doc_id={doc_id} boxLocation={boxLocation} />
        <ExtractedFields doc_id={doc_id} handleClick={changeBox} />
    </div>
  )
}

export default InteractiveSpace