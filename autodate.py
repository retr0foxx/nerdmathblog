# assigns dates to posts.json based on the creation date of the blog file
# windows only, apparently
import json;
import os;

with open("./posts.json", "r") as f:
    posts = json.load(f)

for post in posts:
    if ("url" not in post):
        continue;

    file = post["url"];
    if ("creation_date" not in posts):
        time = os.path.getctime(file);
        post["creation_date"] = round(time);

print(posts);

with open("./posts.json", "w") as f:
    json.dump(posts, f, indent=4);