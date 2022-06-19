import os
import time

import numpy as np
import pandas as pd
from scipy import io

# config
path = "/home/markus/workspace/data/datasets_sources/faces_imdb/"
filename = "imdb.mat"
mat_key = "imdb"  # wiki or imdb
# END - config

start = time.time()

mat = io.loadmat(os.path.join(path, filename))

dob = mat[mat_key][0][0]['dob'][0].astype(int)
photo_taken = mat[mat_key][0][0]['photo_taken'][0].astype(int)
full_path = np.array([x[0] for x in mat[mat_key][0][0]['full_path'][0]])
gender = mat[mat_key][0][0]['gender'][0].astype(int)
name = np.array([x[0] if len(x) > 0 else None for x in mat[mat_key][0][0]['name'][0]])
face_location = [x[0].astype(np.float64) for x in mat[mat_key][0][0]['face_location'][0]]
face_score = mat[mat_key][0][0]['face_score'][0].astype(np.float64)
second_face_score = mat[mat_key][0][0]['second_face_score'][0].astype(np.float64)

# mat[mat_key][0][0]['face_location'][0]
assert not any([len(x) != 4 for x in face_location])
print(f"face_location: every element has 4 points")

assert len(dob) == len(photo_taken) == len(full_path) == len(gender) == len(name) == len(face_location) == len(face_score) == len(
    second_face_score)
print(f"length check success, len is {len(dob)}")

df = pd.DataFrame.from_dict({
    "dob": dob,
    "photo_taken": photo_taken,
    "full_path": full_path,
    # "gender": gender,  # has some invalid values (-9223372036854775808)
    "name": name,
    # "face_location": face_location,  # not required
    "face_score": face_score,
    "second_face_score": second_face_score,
})
print("not saving: gender, face_location")

if len(df[df["name"].isnull()]) > 0:
    print("There samples without names. Thats ok. They are None")

print(f"loaded {len(df)} lines. Saving them to {mat_key}.csv\n"
      f"They contain samples that might already have been/will be deleted (to small filesize)\n")
df.to_csv(os.path.join(path, f"{mat_key}.csv"))

# remove samples with no face
df = df[~df["face_score"].isin([-np.inf, np.nan])]

# remove samples with second face; labels might be for the wrong person
df = df[~(df["second_face_score"] > 0)]
print(f"left after filtering {len(df)} lines. Saving them to {mat_key}_filtered.csv\n"
      f"removed samples with no face and samples with second face on source image\n"
      f"There might still be samples that have already have been/will be deleted (to small filesize)\n")
df.to_csv(os.path.join(path, f"{mat_key}_filtered.csv"))

print(f"took {time.time() - start}s")

"""
for reference: some trial and error of trying to do this with octave. Loading via double click and running some of the lines starting at
"pkg load io" partially worked

#FileData = load("imdb.mat")
#csvwrite("imdb.csv", FileData.imdb)


#a = [struct2cell(imdb){:}]

#writetable(a, 'Structure_Example.csv')



pkg load io
#wiki = rmfield(wiki,"face_location")
#wiki = rmfield(wiki,"name")
#wiki.face_score(isinf(wiki.face_score))=-2
#wiki.second_face_score(isinf(wiki.second_face_score))=-2
#wiki.second_face_score(isnan(wiki.second_face_score))=-1
#wiki.gender(isnan(wiki.gender))=-1
#wiki.face_score(isnan(wiki.face_score))=-1
#c = struct2cell(wiki)
pkg load io



#cell2csv(c, "wiki.csv")



#csvwrite("wiki_dob.csv", wiki.dob)
#csvwrite("wiki_photo_taken.csv", wiki.photo_taken)
#cell2csv("wiki_full_path.csv", wiki.full_path)
#csvwrite("wiki_gender.csv", wiki.gender)
#cell2csv("wiki_name.csv", wiki.name)
cell2csv("wiki_face_location.csv", wiki.face_location(:,0))
#csvwrite("wiki_face_score.csv", wiki.face_score)
#csvwrite("wiki_second_face_score.csv", wiki.second_face_score)
"""
