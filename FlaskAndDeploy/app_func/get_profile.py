# base
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# personality insight
from ibm_watson import PersonalityInsightsV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

def generate_profile_df(profile, element):

    ex_dict = {el['name']:el['percentile'] for el in profile[element]}

    return pd.DataFrame.from_records(ex_dict, index=[0], columns=ex_dict.keys())

def text2profile(text, profile_elements = ['personality', 'needs', 'values']):

    profile = service.profile(
        text,
        'application/json',
        raw_scores=True,
        consumption_preferences=False).get_result()

    result = pd.DataFrame()
    for el in profile_elements:
        result = pd.concat([result, generate_profile_df(profile, el)], axis = 1)

    return result.iloc[0]


if __name__ == '__main__':

    input_text = "EVERY substance is negatively electric to that which stands above \
    in the chemical tables, positively to that which stands below \
    it. Water dissolves wood and iron and salt; air dissolves water;\
    electric fire dissolves air, but the intellect dissolves fire, gravity,\
    laws, method, and the subtlest unnamed relations of nature in its \
    resistless menstruum. Intellect lies behind genius, which is intellect constructive. Intellect is the simple power anterior to all \
    action or construction. Gladly would I unfold in calm degrees a \
    natural history of the intellect, but what man has yet been able \
    to mark the steps and boundaries of that transparent essence? \
    The first questions are always to be asked, and the wisest doctor \
    is gravelled by the inquisitiveness of a child. How can we speak \
    of the action of the mind under any divisions, as of its knowledge, \
    of its ethics, of its works, and so forth, since it melts will into \
    perception, knowledge into act? Each becomes the other. Itself \
    alone is. Its vision is not like the vision of the eye, but is union \
    with the things known.\
    Intellect and intellection signify to the common ear consideration of abstract truth. The consideration of time and place, of \
    you and me, of profit and hurt, tyrannize over most men's minds.\
    Intellect separates the fact considered, from you, from all local\
    and personal reference, and discerns it as if it existed for its own \
    sake. Heraclitus looked upon the affections as dense and colored \
    mists. In the fog of good and evil affections it is hard for man to \
    walk forward in a straight line .. Intellect is void of affection and \
    sees an object as it stands in the light of science, cool and disengaged. The intellect goes out of the individual, floats over its own \
    personality, and regards it as a fact, and not as I and mine. He, \
    who is immersed in what concerns person or place cannot see the \
    problem of existence. This the intellect always ponders. Nature \
    shows all things formed and bound. The intellect pierces the \
    form, overleaps the wall, detects intrinsic likeness between remote things and reduces all things into a few principles."


    url = "https://gateway-fra.watsonplatform.net/personality-insights/api"
    apikey = "w-s1kGzcVV8xeTzYvYgwsIKk4UAF8M2Zr7xkPRgfiKCd"

    # # Authentication via IAM
    authenticator = IAMAuthenticator(apikey)
    service = PersonalityInsightsV3(
        version='2017-10-13',
        authenticator=authenticator)
    service.set_service_url(url)

    profile = text2profile(input_text)

    profile = pd.DataFrame(profile)
    profile = profile.T
    print("Profile: ", profile)
