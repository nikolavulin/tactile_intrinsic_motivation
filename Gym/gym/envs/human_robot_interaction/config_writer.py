import yaml
import io

# Define data
FINGERTIP_SITE_NAMES = [
    'robot1:S_palm',
    'robot1:S_thtip',
    'robot1:S_fftip',
    'robot1:S_mftip',
    'robot1:S_rftip',
    'robot1:S_lftip',
]


ROBOT1_BODY_NAMES = [
    'robot1:palm',
    'robot1:ffknuckle',
    'robot1:ffproximal',
    'robot1:ffmiddle',
    'robot1:ffdistal',
    'robot1:mfknuckle',
    'robot1:mfproximal',
    'robot1:mfmiddle',
    'robot1:mfdistal',
    'robot1:lfknuckle',
    'robot1:lfproximal',
    'robot1:lfmiddle',
    'robot1:lfdistal',
    'robot1:rfknuckle',
    'robot1:rfproximal',
    'robot1:rfmiddle',
    'robot1:rfdistal',
    'robot1:thbase',
    'robot1:thproximal',
    'robot1:thmiddle',
    'robot1:thdistal'
    ]

ROBOT1_JOINT_NAMES = [
    'robot1:WRJ1',
    'robot1:WRJ0',
    'robot1:THJ4',
    'robot1:THJ3',
    'robot1:THJ2',
    'robot1:THJ1',
    'robot1:THJ0',
    'robot1:FFJ3',
    'robot1:FFJ2',
    'robot1:FFJ1',
    'robot1:MFJ3',
    'robot1:MFJ2',
    'robot1:MFJ1',
    'robot1:LFJ4',
    'robot1:LFJ3',
    'robot1:LFJ2',
    'robot1:LFJ1',
    'robot1:RFJ2',
    'robot1:RFJ1',
]

ROBOT0_JOINT_NAMES = [
    'robot0:WRJ1',
    'robot0:WRJ0',
    'robot0:THJ4',
    'robot0:THJ3',
    'robot0:THJ2',
    'robot0:THJ1',
    'robot0:THJ0',
    'robot0:FFJ3',
    'robot0:FFJ2',
    'robot0:FFJ1',
    'robot0:MFJ3',
    'robot0:MFJ2',
    'robot0:MFJ1',
    'robot0:LFJ4',
    'robot0:LFJ3',
    'robot0:LFJ2',
    'robot0:LFJ1',
    'robot0:RFJ3',
    'robot0:RFJ2',
    'robot0:RFJ1',
]


GOAL_SITES = [
    'robot1:thumb_goal',
    'robot1:S_lftip_ff',
    'robot1:S_lftip_mf',
    'robot1:S_lftip_rf',
    'robot1:S_lftip_lf',
]

ROBOT_SITES = [
    'robot0:S_thtip',
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
]

TOUCH_SENSORS = [
    'robot0:ST_Tch_fftip',
]


ROBOT0_CONTROL_JOINTS = [
    'robot0:shoulder_pan_joint',
    'robot0:shoulder_lift_joint',
    'robot0:upperarm_roll_joint',
    'robot0:elbow_flex_joint',
    'robot0:WRJ1',
    'robot0:WRJ0',
    'robot0:FFJ3',
    'robot0:FFJ2',
    'robot0:FFJ1',
    'robot0:FFJ0',
    'robot0:MFJ3',
    'robot0:MFJ2',
    'robot0:MFJ1',
    'robot0:MFJ0',
    'robot0:RFJ3',
    'robot0:RFJ2',
    'robot0:RFJ1',
    'robot0:RFJ0',
    'robot0:LFJ4',
    'robot0:LFJ3',
    'robot0:LFJ2',
    'robot0:LFJ1',
    'robot0:LFJ0',
    'robot0:THJ4',
    'robot0:THJ3',
    'robot0:THJ2',
    'robot0:THJ1',
    'robot0:THJ0',
]

FORCE_GOALS = [1]

data = {'FINGERTIP_SITE_NAMES': FINGERTIP_SITE_NAMES,
        'ROBOT1_BODY_NAMES': ROBOT1_BODY_NAMES,
        'ROBOT1_JOINT_NAMES': ROBOT1_JOINT_NAMES,
        'ROBOT0_JOINT_NAMES': ROBOT0_JOINT_NAMES,
        'GOAL_SITES': GOAL_SITES,
        'ROBOT_SITES': ROBOT_SITES,
        'TOUCH_SENSORS': TOUCH_SENSORS,
        'ROBOT0_CONTROL_JOINTS': ROBOT0_CONTROL_JOINTS,
        'FORCE_GOALS': FORCE_GOALS}

# Write YAML file
with io.open('config/et.yaml', 'w', encoding='utf8') as outfile:
   yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

# with open("config/handshake.yaml", 'r') as stream:
#     data_loaded = yaml.safe_load(stream)
