import json
import numpy as np
import pandas as pd


def timedelta64_to_secs(timedelta):
    """Convert `timedelta` to seconds.

    Parameters
    ----------
    timedelta : np.timedelta64

    Returns
    -------
    float : seconds
    """
    if len(timedelta) == 0:
        return np.array([])
    else:
        return timedelta / np.timedelta64(1, 's')


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return json.JSONEncoder.default(self, obj)


def get_activations(chunk, params):
    """Returns runs of an appliance.

    Most appliances spend a lot of their time off.  This function finds
    periods when the appliance is on.

    Parameters
    ----------
    chunk : pd.Series
    min_off_duration : int
        If min_off_duration > 0 then ignore 'off' periods less than
        min_off_duration seconds of sub-threshold power consumption
        (e.g. a washing machine might draw no power for a short
        period while the clothes soak.)  Defaults to 0.
    min_on_duration : int
        Any activation lasting less seconds than min_on_duration will be
        ignored.  Defaults to 0.
    border : int
        Number of rows to include before and after the detected activation
    on_power_threshold : int or float
        Watts

    Returns
    -------
    list of pd.Series.  Each series contains one activation.
    """
    min_off_duration = params['N_off']
    min_on_duration = params['N_on']
    border = params['border']
    on_power_threshold = params['p']
    when_on = chunk >= on_power_threshold
    # print(chunk)
    state = pd.DataFrame(np.zeros_like(chunk), index=chunk.index)
    # print(state)
    # Find state changes
    state_changes = when_on.astype(np.float32).diff()

    switch_on_events = np.where(state_changes == 1)[0]
    switch_off_events = np.where(state_changes == -1)[0]

    if len(switch_on_events) == 0 or len(switch_off_events) == 0:
        if (when_on[0]):
            state[:] = 1
            return [], state
        else:
            return [], state

    del when_on
    del state_changes

    # Make sure events align
    if switch_off_events[0] < switch_on_events[0]:
        state[:switch_off_events[0]] = 1
        switch_off_events = switch_off_events[1:]
        if len(switch_off_events) == 0:
            return [], state
    if switch_on_events[-1] > switch_off_events[-1]:
        state[switch_on_events[-1]:] = 1
        switch_on_events = switch_on_events[:-1]
        if len(switch_on_events) == 0:
            return [], state
    assert len(switch_on_events) == len(switch_off_events)

    # Smooth over off-durations less than min_off_duration
    if min_off_duration > 0:
        off_durations = (chunk.index[switch_on_events[1:]].values -
                         chunk.index[switch_off_events[:-1]].values)

        off_durations = timedelta64_to_secs(off_durations)

        above_threshold_off_durations = np.where(
            off_durations >= min_off_duration)[0]

        # Now remove off_events and on_events
        switch_off_events = switch_off_events[
            np.concatenate([above_threshold_off_durations,
                            [len(switch_off_events) - 1]])]
        switch_on_events = switch_on_events[
            np.concatenate([[0], above_threshold_off_durations + 1])]
    assert len(switch_on_events) == len(switch_off_events)

    activations = []
    for on, off in zip(switch_on_events, switch_off_events):
        duration = (chunk.index[off] - chunk.index[on]).total_seconds()
        if duration < min_on_duration:
            continue
        on -= 1 + border
        if on < 0:
            on = 0
        off += border
        activation = chunk.iloc[on:off]
        state.iloc[on:off] = 1
        # throw away any activation with any NaN values
        if not activation.isnull().values.any():
            activations.append(activation)

    return activations, state


config = {
    'threshold': {
        'microwave': {'p': 50, 'N_off': 10, 'N_on': 10, 'border': 1},
        'fridge': {'p': 5, 'N_off': 60, 'N_on': 60, 'border': 1},
        'dish washer': {'p': 10, 'N_off': 300, 'N_on': 1800, 'border': 1}
    },
    'result': {
        'MSE': [],
        'MAE': [],
        'ACC': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
        'sMAE': []
    }
}


def get_sections_df(chunk, good_section):
    result = []
    for section in good_section:
        temp = chunk[section.start:section.end]
        if (temp.shape[0] > 1000):
            result.append(temp)
    return result


def get_sections_df_2(main_section, app_section):
    result = []
    index = pd.date_range(start=main_section[0].start, end=main_section[-1].end, freq='s')
    test = pd.DataFrame(index=index)
    test['mains'] = 0
    test['apps'] = 0
    # print('-')

    for sec in main_section:
        test.loc[sec.start:sec.end, 'mains'] = 1
    # print('-')
    for sec in app_section:
        test.loc[sec.start:sec.end, 'apps'] = 1
    # print('-')

    test['all'] = 0
    test['all'] = ((test['mains'] == 1) & (test['apps'] == 1)).astype(int)
    test['start'] = test['all'].diff()
    if test['all'].iloc[0] == 1:
        test['start'].iloc[0] = 1

    test['end'] = test['all'].diff().fillna(100)
    test['end'] = test[['end']].apply(lambda x: x.shift(-1))
    if test['all'].iloc[-1] == 1:
        test['end'].iloc[-1] = -1
    start_index = index[test['start'] == 1]
    end_index = index[test['end'] == -1]

    for i in range(len(start_index)):
        start = start_index[i]
        end = end_index[i]
        if (end - start) / np.timedelta64(1, 's') > 3000:
            result.append((start, end))
    return result
