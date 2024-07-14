import json

train_videos_path = 'train_list_new.txt'
test_videos_path = 'test_list_new.txt'
annotations_path = 'MSR_VTT.json'

train_video_file = open(train_videos_path, "r")
test_video_file = open(test_videos_path, "r")

train_annotations = []
test_annotations = []

for train_video in train_video_file:
  
  cleaned_string = train_video.replace('\n', '')
  train_annotations.append(cleaned_string)
  
for test_video in test_video_file:
  
  cleaned_string = test_video.replace('\n', '')
  test_annotations.append(cleaned_string)

with open(annotations_path, 'r') as file:
    train_video_count = 0
    test_video_count = 0

    data_1 = []
    data_2 = []

    data = json.load(file)
    annotations = data['annotations']
    print(len(annotations))

    for rec in annotations:
      if rec['image_id'] in test_annotations:
        data_test = {'video_id':rec['image_id'], "caption":rec['caption']}
        data_1.append(data_test)
        test_video_count = test_video_count + 1

      if rec['image_id'] in train_annotations:
        data_train = {'video_id':rec['image_id'], "caption":rec['caption']}
        data_2.append(data_train)
        train_video_count = train_video_count + 1
    
    train_annotions_json = {"train_annotations": data_2}
    test_annotions_json = {"test_annotations": data_1}

    file_path_1 = 'train_annotations.json'
    file_path_2 = 'test_annotations.json'

    with open(file_path_1, 'w') as file:
        json.dump(train_annotions_json, file, indent=4)

    with open(file_path_2, 'w') as file:
        json.dump(test_annotions_json, file, indent=4)

print(f"Test annotations count: {test_video_count}")
print(f"Train annotations count: {train_video_count}")
