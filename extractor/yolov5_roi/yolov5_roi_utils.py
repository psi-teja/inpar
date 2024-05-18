class YOLOv5LabelEncoder:
    def __init__(self):
        self.label_to_id = {}
        self.id_to_label = {}
        self.current_id = 0
    
    def add_label(self, label):
        if label not in self.label_to_id:
            self.label_to_id[label] = self.current_id
            self.id_to_label[self.current_id] = label
            self.current_id += 1
    
    def get_label_id(self, label):
        if label in self.label_to_id:
            return self.label_to_id[label]
        else:
            return None
    
    def get_id_label(self, label_id):
        if label_id in self.id_to_label:
            return self.id_to_label[label_id]
        else:
            return None
    
    def get_label_dict(self):
        return self.label_to_id
    
    def get_id_dict(self):
        return self.id_to_label


if __name__ == "__main__":
    # Example usage
    label_encoder = YOLOv5LabelEncoder()

    # Adding labels
    label_encoder.add_label("label1")
    label_encoder.add_label("label2")

    # Getting label dictionary
    label_dict = label_encoder.get_label_dict()
    print(label_dict)  # Output: {'label1': 0, 'label2': 1}

    # Getting ID dictionary
    id_dict = label_encoder.get_id_dict()
    print(id_dict)  # Output: {0: 'label1', 1: 'label2'}

    # Adding a new label
    label_encoder.add_label("label3")

    # Getting updated label dictionary
    label_dict = label_encoder.get_label_dict()
    print(label_dict)  # Output: {'label1': 0, 'label2': 1, 'label3': 2}

    # Getting updated ID dictionary
    id_dict = label_encoder.get_id_dict()
    print(id_dict)  # Output: {0: 'label1', 1: 'label2', 2: 'label3'}
