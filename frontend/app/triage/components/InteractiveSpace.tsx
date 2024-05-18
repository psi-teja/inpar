import { useState } from "react";
import DocViewer from "@/app/triage/components/DocViewer";
import ExtractedFields from "@/app/triage/components/ExtractedFields";

interface InteractiveSpaceProps {
  doc_id: string | null;
}

const InteractiveSpace: React.FC<InteractiveSpaceProps> = ({ doc_id}) => {
  const [boxLocation, setBoxLocation] = useState<Record<string, any> | null>(null);
  const [selectedField, setSelectedField] = useState<string | null>(null);
  const [view, setView] = useState("DocInfo");

  const changeBox = (fieldName: string | null, location: Record<string, any>) => {
    if (location.pageNo !== 0) {
      setBoxLocation(location);
      setSelectedField(fieldName);
    }
  };

  const changeView = (viewType: string) => {
    setView(viewType);
  };

  return (
    <div className={`${view === "DocInfo" ? "flex" : ""}`}>
      <DocViewer doc_id={doc_id} boxLocation={boxLocation} viewType={view} />
      <ExtractedFields
        doc_id={doc_id}
        handleFieldClick={changeBox}
        handleChangeView={changeView}
        viewType={view}
        selectedField={selectedField}
      />
    </div>
  );
};

export default InteractiveSpace;
