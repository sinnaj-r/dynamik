u"""
# What is `epd`?

```epd``` stands for ***Explainable Performance Drift***, an algorithm for detecting changes in the cycle time of a
process execution and, if present, to provide insights about the actionable causes of the change.

You can find more details about the algorithm in the following article \N{notebook}:

<div style="background-color: #EFEFEF; display: flex; flex-direction: column; padding: 1em; position: relative;">
    <p style="
        font-weight: 700;
        font-size: 1.2em;
        margin: 0;
    ">
        Article title
    </p>
    <p style="
        font-style: italic;
        font-size: .85em;
        font-weight: 500;
        color: #A0A0A0;
        margin: 0;
    ">
        Authors
    </p>
    <p style="
        margin: 0;
        font-size: .85em;
        font-weight: 400;
        color: #A0A0A0;
    ">
        Publisher
    </p>

    <button onclick="navigator.clipboard.writeText('bibtex aqui')"
            style="
                position: absolute;
                top: 12px;
                right: 12px;
                border: 0;
                padding: .25em;
            "
    >
        <svg fill="#A0A0A0"
            height="24px"
            width="24px"
            viewBox="0 0 64 64"
            stroke="#A0A0A0"
            stroke-width="1.25">
                <path d="M53.9791489,9.1429005H50.010849c-0.0826988,0-0.1562004,0.0283995-0.2331009,0.0469999V5.0228 C49.7777481,2.253,47.4731483,0,44.6398468,0h-34.422596C7.3839517,0,5.0793519,2.253,5.0793519,5.0228v46.8432999 c0,2.7697983,2.3045998,5.0228004,5.1378999,5.0228004h6.0367002v2.2678986C16.253952,61.8274002,18.4702511,64,21.1954517,64 h32.783699c2.7252007,0,4.9414978-2.1725998,4.9414978-4.8432007V13.9861002 C58.9206467,11.3155003,56.7043495,9.1429005,53.9791489,9.1429005z M7.1110516,51.8661003V5.0228 c0-1.6487999,1.3938999-2.9909999,3.1062002-2.9909999h34.422596c1.7123032,0,3.1062012,1.3422,3.1062012,2.9909999v46.8432999 c0,1.6487999-1.393898,2.9911003-3.1062012,2.9911003h-34.422596C8.5049515,54.8572006,7.1110516,53.5149002,7.1110516,51.8661003z M56.8888474,59.1567993c0,1.550602-1.3055,2.8115005-2.9096985,2.8115005h-32.783699 c-1.6042004,0-2.9097996-1.2608986-2.9097996-2.8115005v-2.2678986h26.3541946 c2.8333015,0,5.1379013-2.2530022,5.1379013-5.0228004V11.1275997c0.0769005,0.0186005,0.1504021,0.0469999,0.2331009,0.0469999 h3.9682999c1.6041985,0,2.9096985,1.2609005,2.9096985,2.8115005V59.1567993z"></path>
                <path d="M38.6031494,13.2063999H16.253952c-0.5615005,0-1.0159006,0.4542999-1.0159006,1.0158005 c0,0.5615997,0.4544001,1.0158997,1.0159006,1.0158997h22.3491974c0.5615005,0,1.0158997-0.4542999,1.0158997-1.0158997 C39.6190491,13.6606998,39.16465,13.2063999,38.6031494,13.2063999z"></path>
                <path d="M38.6031494,21.3334007H16.253952c-0.5615005,0-1.0159006,0.4542999-1.0159006,1.0157986 c0,0.5615005,0.4544001,1.0159016,1.0159006,1.0159016h22.3491974c0.5615005,0,1.0158997-0.454401,1.0158997-1.0159016 C39.6190491,21.7877007,39.16465,21.3334007,38.6031494,21.3334007z"></path>
                <path d="M38.6031494,29.4603004H16.253952c-0.5615005,0-1.0159006,0.4543991-1.0159006,1.0158997 s0.4544001,1.0158997,1.0159006,1.0158997h22.3491974c0.5615005,0,1.0158997-0.4543991,1.0158997-1.0158997 S39.16465,29.4603004,38.6031494,29.4603004z"></path>
                <path d="M28.4444485,37.5872993H16.253952c-0.5615005,0-1.0159006,0.4543991-1.0159006,1.0158997 s0.4544001,1.0158997,1.0159006,1.0158997h12.1904964c0.5615025,0,1.0158005-0.4543991,1.0158005-1.0158997 S29.0059509,37.5872993,28.4444485,37.5872993z"></path>
        </svg>
    </button>
</div>


# Quickstart
`epd` is designed to be executed using an event log as it's input, processing it sequentially, event by event, mimicking
an online environment where events are consumed from an event stream.

## \N{Old Personal Computer} Running `epd` from the CLI
To run the algorithm in your command line, you need an event log in CSV or JSON format.

If in CSV, the file is expected to have a headers row which is followed by a row per event:
```
case,                   start,                     end,   activity,        resource
2182, 2023-02-24T05:28:00.843, 2023-02-24T05:28:00.843,      START, resource-000001
2182, 2023-02-24T05:28:00.843, 2023-02-24T05:34:31.219, Activity 1, resource-000044
2182, 2023-02-24T05:34:31.219, 2023-02-24T05:47:25.817, Activity 2, resource-000024
2182, 2023-02-24T05:47:25.817, 2023-02-24T05:59:46.195, Activity 3, resource-000010
2182, 2023-02-24T05:59:46.193, 2023-02-24T05:59:46.193,        END, resource-000001
7897, 2023-03-01T08:39:42.861, 2023-03-01T08:39:42.861,      START, resource-000001
7897, 2023-03-01T08:39:42.861, 2023-03-01T08:53:41.167, Activity 1, resource-000029
7897, 2023-03-01T08:53:41.167, 2023-03-01T08:56:46.299, Activity 2, resource-000007
7897, 2023-03-01T08:56:46.299, 2023-03-01T09:12:49.468, Activity 3, resource-000018
7897, 2023-03-01T09:12:49.468, 2023-03-01T09:12:49.468,        END, resource-000001
  ...                      ...                      ...         ...             ...
```
In the case of a JSON file, an array of objects is expected, where each object contains an event:
```
[
    {
        "case": "2182", "activity": "Activity 3", "resource": "resource-000001",
        "start: "2023-02-24T05:28:00.843", "end": "2023-02-24T05:28:00.843"
    },
    {
        "case": "2182", "activity": "Activity 1", "resource": "resource-000044",
        "start: "2023-02-24T05:28:00.843", "end": "2023-02-24T05:34:31.219"
    },
    {
        "case": "2182", "activity": "Activity 2", "resource": "resource-000024",
        "start: "2023-02-24T05:34:31.219", "end": "2023-02-24T05:47:25.817"
    }
    ...
]
```

If your log files have any of these formats you can run `edm` specifying the log format via the option `--format`:
```shell
$> edp ./log.csv --format csv
```
or
```shell
$> edp ./log.json --format json
```

If you need a different mapping for processing your log files, you can specify it using a JSON file:
```
{
    "case": <your case attribute name>,
    "activity": <your activity attribute name>,
    "resource": <your resource attribute name>,
    "start": <your start time attribute name>,
    "end": <your end time attribute name>
}
```
Then, you can then run `epd` with the option `--mapping MAPPING_FILE` to use your custom mapping:
```shell
$> edp ./log.csv --format csv --mapping ./my_custom_mapping.json
```
Run `epd --help` to check the additional options:
```
edp --help

usage: epd [-h] [-f FORMAT] [-m MAPPING_FILE] [-t TIMEFRAME] [-a ALPHA] [-v] LOG_FILE

Explainable Performance Drift is an algorithm for finding actionable causes for drifts in the
performance of a process execution. For this, the cycle time of the process is monitored, and,
if a change is detected in the process performance, the algorithm finds the actionable causes
for the change.

positional arguments:
  LOG_FILE                                    The event log, in CSV or JSON format.

options:
  -h, --help                                  show this help message and exit
  -f FORMAT, --format FORMAT                  specify the event log format
  -m MAPPING_FILE, --mapping MAPPING_FILE     provide a custom mapping file
  -t TIMEFRAME, --timeframe TIMEFRAME         provide a timeframe size, in days, used
                                              to define the reference and running models.
  -a ALPHA, --alpha ALPHA                     specify the confidence for the statistical tests
  -v, --verbose                               enable verbose output. High verbosity level can
                                              drastically decrease edp performance

epd is licensed under the Apache License, Version 2.0

```

## \N{snake} Using `epd` as a Python package
Aside of providing an executable command, `epd` can be fully customized by using it as a Python package.
If you use poetry you can install `edp` directly from git:

```shell
$> poetry add "https://gitlab.citius.usc.es/ProcessMining/explainable-performance-drift.git"

```

When using it as a package, the drift detection algorithm can be located at `epd.drift.detect_drift`.

# How can I...?
## ...read a log from different source than CSV or JSON?
`epd` provides you with all the utilities needed to read CSV and JSON files in a performant way.
However, we are aware that your logs may be in a different format. You may even want to read your logs from a database
or message queue! Don't worry, we've got you covered. The representation we use for logs in epd is a simple `Iterator`
object, so you can implement any data source you need, and as long as it returns an `Iterator` of `epd.model.Event`
objects you shouldn't have any problems. As a reference, you can check the implementations from `epd.input.csv.read_csv_log`
and `epd.input.json.read_json_log`, which are implemented using generators so the memory consumption is reduced.

## ...check the performance for only a part of my process?
In `epd` you can specify exactly which activities mark the start and the end of the subprocess that you want to monitor.
To do that, you just have to specify your initial and final activities when calling `epd.drift.detect_drift`, and the
algorithm will deal with the rest!
"""

__all__ = ["drift", "input", "model"]
__docformat__ = "markdown"
