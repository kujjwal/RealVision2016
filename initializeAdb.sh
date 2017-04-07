#!/bin/bash

adb reverse tcp:5805 tcp:5805
adb shell input keyevent 82
adb shell monkey -p com.example.ujjwal.realvision2016 -c android.intent.category.LAUNCHER 1
