function initMap() {
  var kathleen = {lat: 37.2562, lng: -122.0370};
  var shereen = {lat: 37.3060, lng: -121.9506};
  var anne = {lat: 37.5490, lng: -121.9824};
  var williams = {lat: 37.3770, lng: -121.9259};
  var rayna = {lat: 37.2631, lng: -122.0384};

  var icons = 'http://maps.google.com/mapfiles/ms/icons/blue-dot.png';

  var map = new google.maps.Map(document.getElementById('map'), {
    zoom: 9.65,
    center: shereen
  });

  var contentString1 = '<div id="content">'+
      '<div id="siteNotice">'+
      '</div>'+
      '<h1 id="firstHeading" class="firstHeading">Kathleen Russ: MA</h1>'+
      '<div id="bodyContent">'+
      '<p><b>Type:</b> Psychotherapist' +
      '<p><b>Availability:</b> 6/20/2020-6/30/2020, 10:00am - 6pm'+
      '<p><b>Bio Message:</b> I specialize in Jungian Psychotherapy, Aging & Caregiving, and Expressive Arts in my Therapy Treatments'+
    '</div>'+
      '</div>';

      var infowindow1 = new google.maps.InfoWindow({
        content: contentString1
      });

      var marker1 = new google.maps.Marker({
        position: kathleen,
        map: map,
      });

      marker1.addListener('click', function() {
        infowindow1.open(map, marker1);
      });

      var contentString2 = '<div id="content">'+
          '<div id="siteNotice">'+
          '</div>'+
          '<h1 id="firstHeading" class="firstHeading">Shereen Mohsen, Psy.D.</h1>'+
          '<div id="bodyContent">'+
          '<p><b>Type: </b>  Licensed Clinical Psychologist:' +
          '<p><b>Preferred Dates:</b> 6/30/2020-7/7/2020, 9:00am - 5:30pm'+
          '<p><b>Bio Message:</b> I work on Individual Counseling, Marriage and Family, Consultation, Health and Wellness, Crisis Intervention / Trauma, and Education / Schooling'+
        '</div>'+
          '</div>';

          var infowindow2 = new google.maps.InfoWindow({
            content: contentString2
          });

          var marker2 = new google.maps.Marker({
            position: shereen,
            map: map,

          });

          marker2.addListener('click', function() {
            infowindow2.open(map, marker2);
          });

          var contentString3 = '<div id="content">'+
          '<div id="siteNotice">'+
          '</div>'+
          '<h1 id="firstHeading" class="firstHeading">Anne Bisek, Psy.D.</h1>'+
          '<div id="bodyContent">'+
          '<p><b>Type: </b> Licensed Clinical Psychologist:' +
          '<p><b>Preferred Dates:</b> 6/25/2020-7/10/2020, 9:00am - 5:30pm'+
          '<p><b>Bio Message:</b> I work primarily with first responders, communications, firefighters, pre-hospital care personnel, law enforcement and military veterans.'+
        '</div>'+
          '</div>';

              var infowindow3 = new google.maps.InfoWindow({
                content: contentString3
              });

              var marker3 = new google.maps.Marker({
                position: anne,
                map: map,

              });

              marker3.addListener('click', function() {
                infowindow3.open(map, marker3);
              });


              var contentString4 = '<div id="content">'+
              '<div id="siteNotice">'+
              '</div>'+
              '<h1 id="firstHeading" class="firstHeading">Martin H. Williams, Ph.D.</h1>'+
              '<div id="bodyContent">'+
              '<p><b>Type: </b> Licensed Clinical Psychologist, American Psychological Association:' +
              '<p><b>Preferred Dates:</b> 6/25/2020-7/10/2020, 9:00am - 5:30pm'+
              '<p><b>Bio Message:</b> I do evaluations of emotional damage deriving from personal injury, sexual or racial harassment and sexual abuse (including in psychotherapy) using objective psychological assessment devices'+
            '</div>'+
              '</div>';

                  var infowindow4 = new google.maps.InfoWindow({
                    content: contentString4
                  });

                  var marker4 = new google.maps.Marker({
                    position: williams,
                    map: map,

                  });

                  marker4.addListener('click', function() {
                    infowindow4.open(map, marker4);
                  });

                  var contentString5 = '<div id="content">'+
                  '<div id="siteNotice">'+
                  '</div>'+
                  '<h1 id="firstHeading" class="firstHeading">Rayna Lumbard: LMFT</h1>'+
                  '<div id="bodyContent">'+
                  '<p><b>Type: </b> Licensed Clinical Psychologist' +
                  '<p><b>Preferred Dates:</b> 6/22/2020-7/5/2020, 11:00am - 7:30pm'+
                  '<p><b>Bio Message:</b> I provide the tools to raise your self=worth, your InnerSuccess, the foundation to accomplish your goals and dreams in your relationships, career, health and finances. '+
                '</div>'+
                  '</div>';

                      var infowindow5 = new google.maps.InfoWindow({
                        content: contentString5
                      });

                      var marker5 = new google.maps.Marker({
                        position: rayna,
                        map: map,

                      });

                      marker5.addListener('click', function() {
                        infowindow5.open(map, marker5);
                      });


}
