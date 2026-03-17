from pyinstrument import Profiler

profiler = Profiler()
profiler.start()

# your code here
Crop2()

profiler.stop()
print(profiler.output_text(unicode=True, color=True))
