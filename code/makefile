CXX=g++
#CPPFLAGS= -d
CPPFLAGS= -O3
#LIBSEARCHPATH=-L/opt/local/lib/
#LIBS=-l SDL2

OBJECTS=$(PROJECT).c
PROJECT=HHsimulator

$(PROJECT) : $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(OBJECTS) -o $@ $(LIBSEARCHPATH) $(LIBS)

.cpp.o :
	$(CXX) -c $(CPPFLAGS) -o $@ $<

clean :
	rm -f $(OBJECTS)
