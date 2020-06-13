function initMap() {
  var saratoga = {lat: 37.2664, lng: -122.0296};
  var gunn = {lat: 37.4025, lng: -122.1334};
  var mv = {lat: 37.3150, lng: -122.0562};
  var jl = {lat: 37.2924, lng: -122.0011};
  var prospect = {lat: 37.5899, lng: -122.0269};

  var icons = 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png';

  var map = new google.maps.Map(document.getElementById('map'), {
    zoom: 10,
    center: saratoga
  });

  var contentString1 = '<div id="content">'+
      '<div id="siteNotice">'+
      '</div>'+
      '<h1 id="firstHeading" class="firstHeading">Saratoga High School</h1>'+
      '<div id="bodyContent">'+
      '<p><b>Excess Device Type:</b> Chromebooks' +
      '<p><b>Excess Device Count:</b> 2600'+
      '<p><b>Preferred Dates:</b> 6/5/2020-6/13/2020'+
      '<p><b>Bio Message:</b> Willing to work with any Bay Area Schools in need of devices.'+
    '</div>'+
      '</div>';

      var infowindow1 = new google.maps.InfoWindow({
        content: contentString1
      });

      var marker1 = new google.maps.Marker({
        position: saratoga,
        map: map,
      });

      marker1.addListener('click', function() {
        infowindow1.open(map, marker1);
      });
/*
      var contentString2 = '<div id="content">'+
          '<div id="siteNotice">'+
          '</div>'+
          '<h1 id="firstHeading" class="firstHeading">Gunn High School</h1>'+
          '<div id="bodyContent">'+
          '<p><b>Excess Device Type:</b> MacBooks' +
          '<p><b>Excess Device Count:</b> 1500'+
          '<p><b>Preferred Dates:</b> 6/12/2020-6/20/2020'+
          '<p><b>Bio Message:</b> Excited to work with any Bay Area Schools to help students.'+
        '</div>'+
          '</div>';
          var infowindow2 = new google.maps.InfoWindow({
            content: contentString2
          });
          var marker2 = new google.maps.Marker({
            position: gunn,
            map: map,
            icon: icons
          });
          marker2.addListener('click', function() {
            infowindow2.open(map, marker2);
          });
          var contentString3 = '<div id="content">'+
              '<div id="siteNotice">'+
              '</div>'+
              '<h1 id="firstHeading" class="firstHeading">Monta Vista High School</h1>'+
              '<div id="bodyContent">'+
              '<p><b>Excess Device Type:</b> iPads' +
              '<p><b>Excess Device Count:</b> 1950'+
              '<p><b>Preferred Dates:</b> 5/30/2020-6/1/2020'+
              '<p><b>Bio Message:</b> If any schools need more devices for their students, please do not hesistate to reach out.'+
            '</div>'+
              '</div>';
              var infowindow3 = new google.maps.InfoWindow({
                content: contentString3
              });
              var marker3 = new google.maps.Marker({
                position: mv,
                map: map,
                icon: icons
              });
              marker3.addListener('click', function() {
                infowindow3.open(map, marker3);
              });
              var contentString4 = '<div id="content">'+
                  '<div id="siteNotice">'+
                  '</div>'+
                  '<h1 id="firstHeading" class="firstHeading">James Logan High School</h1>'+
                  '<div id="bodyContent">'+
                  '<p><b>Device Type Needed:</b> Any' +
                  '<p><b>Device Count Needed:</b> 1400'+
                  '<p><b>Preferred Dates:</b> 6/10/10-6/16/2020'+
                  '<p><b>Bio Message:</b> Would appreciate any support, open to any devices for our students.'+
                '</div>'+
                  '</div>';
                  var infowindow4 = new google.maps.InfoWindow({
                    content: contentString4
                  });
                  var marker4 = new google.maps.Marker({
                    position: jl,
                    map: map,
                    title: 'Uluru (Ayers Rock)'
                  });
                  marker4.addListener('click', function() {
                    infowindow4.open(map, marker4);
                  });
                  var contentString5 = '<div id="content">'+
                      '<div id="siteNotice">'+
                      '</div>'+
                      '<h1 id="firstHeading" class="firstHeading">Prospect High School</h1>'+
                      '<div id="bodyContent">'+
                      '<p><b>Device Type Needed:</b> Chromebooks' +
                      '<p><b>Device Count Needed:</b> 2100'+
                      '<p><b>Preferred Dates:</b> 6/7/10-6/20/2020'+
                      '<p><b>Bio Message:</b> If any schools have excess Chromebooks that would be great.'+
                    '</div>'+
                      '</div>';
                      var infowindow5 = new google.maps.InfoWindow({
                        content: contentString5
                      });
                      var marker5 = new google.maps.Marker({
                        position: prospect,
                        map: map,
                        title: 'Uluru (Ayers Rock)'
                      });
                      marker5.addListener('click', function() {
                        infowindow5.open(map, marker5);
                      });
*/
}
