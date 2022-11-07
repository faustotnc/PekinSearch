from flask import Flask, jsonify

APP = Flask(__name__)


@APP.route("/search/<start>/<end>/<tags>/")
def hello_world(start: str, end: str, tags: str):
    # Convert the timeline dates to numerical format: [MM, DD, YYYY]
    date_start = [int(start[2:4]), int(start[0:2]), int(start[4:8])]
    date_end = [int(end[2:4]), int(end[0:2]), int(end[4:8])]

    # Perform minimal cleaning of the requested tags
    tags = [tag.strip() for tag in tags.strip().split(",") if len(tag.strip()) > 0]

    return jsonify({
        "metadata": {
            "path_name": "search query request",
            "queried_tags": tags,
            "timeline": {
                "start": {
                    "month": date_start[0],
                    "day": date_start[1],
                    "year": date_start[2]
                },
                "end": {
                    "month": date_end[0],
                    "day": date_end[1],
                    "year": date_end[2]
                }
            }
        },
        "posts": []
    })
