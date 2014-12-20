import json
import sys

__author__ = 'Ilya Markov'


class Session(object):
    """
    Represents a search session.
    """
    def __init__(self, query):
        """
        Initializes a search session for a given query.
        """
        self.query = query

        self.web_results = []
        self.vertical_blocks = []

    def add_web_result(self, web_result):
        self.web_results.append(web_result)

    def add_vertical_block(self, vertical_block):
        self.vertical_blocks.append(vertical_block)

    def get_clicks(self):
        clicks = [result.click for result in self.web_results]
        return clicks

    def get_last_click_rank(self):
        clicks = self.get_clicks()
        click_ranks = [r for r, click in enumerate(clicks) if click]
        last_click_rank = click_ranks[-1] if len(click_ranks) else len(clicks) - 1
        return last_click_rank

    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    @classmethod
    def from_JSON(cls, json_str):
        session = cls("", "", 0)
        session.__dict__ = json.loads(json_str)

        web_results = []
        for web_result_json in session.web_results:
            web_result = Result.from_JSON(web_result_json)
            web_results.append(web_result)
        session.web_results = web_results

        vert_results = []
        for vert_result_json in session.vertical_blocks:
            vert_result = VerticalBlock.from_JSON(vert_result_json)
            vert_results.append(vert_result)
        session.vertical_blocks = vert_results

        return session

    def __str__(self):
        return self.to_JSON()

    def __repr__(self):
        return str(self)


class VerticalBlock(object):
    """
    Represents a result list from a vertical search engine.
    Includes the vertical name, the position of a result block and a list of result objects.
    """
    def __init__(self, name, position, intent_prob):
        self.name = name
        self.position = position
        self.intent_prob = intent_prob
        self.results = []

    def add_result(self, result):
        self.results.append(result)

    @classmethod
    def from_JSON(cls, json_str):
        vert_result = cls("", 0, 0)
        vert_result.__dict__ = json_str

        results = []
        for result_json in vert_result.results:
            result = Result.from_JSON(result_json)
            results.append(result)
        vert_result.results = results

        # vert_result.position = int(vert_result.position)
        # vert_result.intent_prob = float(vert_result.intent_prob)

        return vert_result


class Result(object):
    """
    Represents an object on a SERP.
    Includes an object id, relevance and click.
    """
    def __init__(self, object, relevance, click):
        self.object = object
        self.relevance = relevance
        self.click = click

    @classmethod
    def from_JSON(cls, json_str):
        result = cls("", 0.0, 0, -1)
        result.__dict__ = json_str

        result.relevance = float(result.relevance)
        # result.click = int(result.click)
        # result.click_time = long(result.click_time)

        return result


# if __name__ == "__main__":
#     input_reader = InputReader()
#     sessions_old, tmp = input_reader.decode(open("../data/train_data"))
#     sessions_new = []
#
#     for session_old in sessions_old:
#         session_new = Session(session_old.uid, session_old.query, session_old.region)
#
#         for indx, url in enumerate(session_old.urls):
#             result = Result(url, session_old.relevances[indx], session_old.clicks[indx], session_old.times[indx])
#             session_new.add_web_result(result)
#
#         vertical_result = VerticalResult("images", session_old.vert_pos)
#         vertical_result.add_result(Result("image", 0.0, session_old.vert_click, session_old.vert_time))
#         session_new.add_vertical_result(vertical_result)
#
#         sessions_new.append(session_new)
#
#     for session in sessions_new:
#         session_json = session.to_JSON()
#         print "JSON", session_json
#
#         session_decoded = Session.from_JSON(session_json)
#         print "DECD", session_decoded

