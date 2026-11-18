// A picture of a project's video (a scanned frame, a crop of one, a preview). mmPicture and mmPublic are what the
// developer hooks' /grab reads: it greys every picture whose project isn't public domain (AGENTS.md, Privacy).
//
// Every scaled-down Image in the app sets sourceSize to the size it is drawn at (x Screen.devicePixelRatio): Qt then
// decodes it at that size with its smooth (area-averaging) scaler, and the GPU draws it about 1:1. Left to the GPU,
// a big reduction (a 4K frame in a 300 px crop, the 824 px icon in 168 px) skips most source pixels and looks jagged;
// `mipmap: true` didn't help here.
import QtQuick

Item {
    property bool mmPicture: true
    property bool mmPublic: false
}
