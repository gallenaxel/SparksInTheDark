lazy val core = (project in file("core"))
	.dependsOn(native % Runtime)

lazy val native = (project in file("native"))
	.enablePlugins(JniNative)
